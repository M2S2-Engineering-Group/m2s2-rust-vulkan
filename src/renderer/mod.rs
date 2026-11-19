pub mod allocator;
pub mod buffer;
mod cleanup;
pub mod command;
pub mod depth;
pub mod device;
pub mod diagnostics;
#[cfg(test)]
pub(crate) mod golden;
mod handle;
#[cfg(test)]
pub(crate) mod headless;
mod image;
pub mod instance;
pub(crate) mod instance_buffer;
pub mod light;
pub mod mesh;
pub(crate) mod mip;
pub(crate) mod per_frame_uniform;
pub mod pipeline;
pub mod scene;
pub(crate) mod submission;
pub mod swapchain;
pub mod text;
pub mod texture;
pub mod uniform;
mod upload;

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use ash::vk;
use command::{ObjectCommand, FRAMES_IN_FLIGHT};
use instance_buffer::{InstanceBuffer, MAX_INSTANCES_PER_FRAME};
use light::{validate_point_lights, DirectionalLight, Light, PointLight};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;
use submission::SubmitState;

/// A retained window/display owner the renderer keeps alive for its whole lifetime, so the safe
/// `with_window` constructor upholds the surface's window-lifetime contract by construction. Blanket-
/// implemented for anything exposing both handle traits (e.g. a winit `Window`).
pub(crate) trait WindowOwner: HasWindowHandle + HasDisplayHandle {}
impl<T: HasWindowHandle + HasDisplayHandle> WindowOwner for T {}
use scene::Object;
use std::mem::ManuallyDrop;
use texture::{validate_specular, SpecularMaterial, StoredTexture};

/// The Vulkan renderer. Owns the whole GPU resource graph; its fields and low-level owners are
/// private. Construct it with [`VulkanRenderer::with_window`] (safe) or [`VulkanRenderer::new`]
/// (unsafe, raw handles).
///
/// External code cannot construct the low-level Vulkan owners:
/// ```compile_fail
/// let _ = m2s2_rust_vulkan::renderer::instance::VulkanInstance::new();
/// ```
///
/// External code cannot read the renderer's private fields:
/// ```compile_fail
/// fn peek(r: &m2s2_rust_vulkan::VulkanRenderer) {
///     let _ = &r.device;
/// }
/// ```
///
/// The raw-handle constructor is `unsafe` and cannot be called without an `unsafe` block:
/// ```compile_fail
/// fn make(d: m2s2_rust_vulkan::RawDisplayHandle, w: m2s2_rust_vulkan::RawWindowHandle) {
///     let _ = m2s2_rust_vulkan::VulkanRenderer::new(d, w, (1, 1));
/// }
/// ```
pub struct VulkanRenderer {
    // `ManuallyDrop` and torn down explicitly in `Drop`, in swapchain -> device -> instance
    // order, instead of relying on Rust's default field-drop order. That default drops struct
    // fields top-to-bottom (declaration order) -- the opposite of what's needed here, since
    // Vulkan requires every object be destroyed before the VkDevice/VkInstance it was created
    // from. See the `Drop` impl below.
    pub(crate) instance: ManuallyDrop<instance::VulkanInstance>,
    pub(crate) device: ManuallyDrop<device::VulkanDevice>,
    pub(crate) swapchain: ManuallyDrop<swapchain::VulkanSwapchain>,
    // Constructed after swapchain; torn down before it in Drop (reverse construction order), so
    // the full destroy chain is frame_context -> pipeline -> swapchain -> device -> instance.
    pub(crate) pipeline: ManuallyDrop<pipeline::GraphicsPipeline>,
    pub(crate) frame_context: ManuallyDrop<command::FrameContext>,
    // Per-frame camera uniforms + descriptor sets, and the depth attachment. Like the buffers (and for
    // the same reason — freeing their allocations needs `&mut Allocator`), these are NOT
    // `ManuallyDrop`; they're torn down via explicit `destroy` in `Drop`, before the allocator.
    // `depth` is also recreated on resize (it's swapchain-extent-sized); `uniforms` is not.
    pub(crate) uniforms: Option<uniform::Uniforms>,
    pub(crate) depth: Option<depth::DepthResources>,
    pub(crate) texture_set_layout: vk::DescriptorSetLayout,
    pub(crate) white_texture: Option<StoredTexture>,
    pub(crate) white_set: vk::DescriptorSet,
    pub(crate) textures: Vec<StoredTexture>,
    pub(crate) texture_sets: Vec<vk::DescriptorSet>,
    // Caller-uploaded meshes (via `create_mesh`), mirroring `textures` above: freed via explicit
    // `destroy` in `Drop` (needs `&mut Allocator`), before the allocator.
    pub(crate) meshes: Vec<mesh::StoredMesh>,
    // This renderer instance's owner id (see `handle.rs`), stamped into every `TextureId`/`MeshId`
    // it hands out so a handle from a different renderer is rejected rather than silently resolved.
    owner_id: u64,
    // Per-frame light uniform (descriptor set 2) and the current directional light it uploads each
    // frame. Like `uniforms`, freed via explicit `destroy` in `Drop` before the allocator.
    pub(crate) light: Option<Light>,
    pub(crate) directional_light: DirectionalLight,
    pub(crate) point_lights: Vec<PointLight>,
    // Per-frame-in-flight instance buffer (plan 016): one `InstanceData` per object, written each
    // frame in `draw_frame` and bound once before the per-object draw loop. Like `light`/`uniforms`,
    // freed via explicit `destroy` in `Drop` before the allocator.
    pub(crate) instance_buffer: Option<InstanceBuffer>,
    // Terminal when a fence wait could not confirm GPU completion: the renderer then leaks its whole
    // resource graph in `Drop` and every public operation returns `VulkanError::Terminal`.
    submit_state: SubmitState,
    pub(crate) allocator: ManuallyDrop<allocator::Allocator>,
    // Retained window/display owner from the safe `with_window` constructor (None for the unsafe
    // raw-handle `new`). Kept alive for the renderer's whole lifetime so the surface's window stays
    // valid through teardown; dropped after the manual teardown in `Drop` has destroyed the surface.
    _window: Option<Arc<dyn WindowOwner>>,
    // A pending resize target in physical pixels; the next `render_frame` recreates the swapchain to
    // it. Set by `resize` and by an out-of-date frame. `None` = nothing pending.
    pending_resize: Option<(u32, u32)>,
    text_renderer: Option<text::TextRenderer>,
    pipeline_formats: (vk::Format, vk::Format),
    text_vertices: Vec<text::TextVertex>,
    device_name: String,
    scene_draw_calls: usize,
    submitted_indices: u64,
    scene_instances: usize,
}

impl VulkanRenderer {
    /// Safe constructor: the renderer retains shared ownership of `window`, so the surface's
    /// window/display connection stays valid for the renderer's whole lifetime — including teardown —
    /// with no caller `unsafe`. `window` provides both handles (e.g. an `Arc<winit::window::Window>`);
    /// keep your own clone for the event loop. `window_extent` is the surface size in physical pixels.
    pub fn with_window<W: HasWindowHandle + HasDisplayHandle + 'static>(
        window: Arc<W>,
        window_extent: (u32, u32),
    ) -> Result<Self> {
        let display_handle = window
            .display_handle()
            .map_err(|e| VulkanError::ValidationError(format!("display handle unavailable: {e}")))?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|e| VulkanError::ValidationError(format!("window handle unavailable: {e}")))?
            .as_raw();
        Self::init(
            display_handle,
            window_handle,
            window_extent,
            Some(window as Arc<dyn WindowOwner>),
        )
    }

    /// Unsafe constructor from raw handles, for consumers that own their own window/event loop and do
    /// not want the renderer to retain a window reference.
    ///
    /// # Safety
    /// `display_handle`/`window_handle` must describe a valid, matching window/display connection, and
    /// their owners must stay alive until the renderer has completed GPU use and destroyed its surface
    /// (including the terminal-leak path). The caller must satisfy platform thread-affinity rules.
    /// Prefer [`VulkanRenderer::with_window`], which upholds this by construction.
    ///
    /// `window_extent` is the surface size in physical pixels. Geometry is uploaded separately via
    /// `create_mesh` once the renderer exists.
    pub unsafe fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
    ) -> Result<Self> {
        Self::init(display_handle, window_handle, window_extent, None)
    }

    /// Shared construction. `window` is the retained owner for the safe path (`None` for raw handles).
    fn init(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
        window: Option<Arc<dyn WindowOwner>>,
    ) -> Result<Self> {
        log::info!("Initializing Vulkan renderer");

        let instance = instance::VulkanInstance::new()?;
        let device = device::VulkanDevice::new(&instance, display_handle, window_handle)?;
        let swapchain = swapchain::VulkanSwapchain::new(
            &instance,
            &device,
            display_handle,
            window_handle,
            window_extent,
        )?;

        // Allocator for the depth/uniform/texture/light resources below; caller geometry is uploaded
        // separately via `create_mesh` once the renderer exists.
        let mut allocator = allocator::Allocator::new(&instance, &device)?;
        let mut depth = None;
        let mut uniforms = None;
        let mut texture_set_layout = None;
        let mut white_texture = None;
        let mut light = None;
        let mut instance_buffer = None;
        let resources = (|| {
            // Depth attachment (swapchain-sized) and per-frame camera uniforms. The uniforms' descriptor-set
            // layout must exist before the pipeline, which references it in its pipeline layout.
            depth = Some(depth::DepthResources::new(
                &instance,
                &device,
                &mut allocator,
                swapchain.extent,
            )?);
            uniforms = Some(uniform::Uniforms::new(&device, &mut allocator)?);

            let layout = texture::create_set_layout(&device)?;
            texture_set_layout = Some(layout);
            white_texture = Some(StoredTexture::new(
                &device,
                &mut allocator,
                &instance,
                layout,
                &[255, 255, 255, 255],
                1,
                1,
                texture::TextureOptions::default(),
            )?);

            light = Some(Light::new(&device, &mut allocator)?);
            instance_buffer = Some(InstanceBuffer::new(
                &device,
                &mut allocator,
                FRAMES_IN_FLIGHT,
            )?);

            let pipeline = pipeline::GraphicsPipeline::new(
                &device,
                swapchain.format,
                depth.as_ref().unwrap().format(),
                &[
                    uniforms.as_ref().unwrap().descriptor_set_layout(),
                    layout,
                    light.as_ref().unwrap().descriptor_set_layout(),
                ],
            )?;
            let frame_context = command::FrameContext::new(&device, swapchain.images.len())?;

            Ok((pipeline, frame_context))
        })();
        let (pipeline, frame_context) = match resources {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(resource) = instance_buffer.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = light.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = white_texture.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(layout) = texture_set_layout.take() {
                    unsafe { device.device.destroy_descriptor_set_layout(layout, None) };
                }
                if let Some(resource) = uniforms.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = depth.take() {
                    resource.destroy(&device, &mut allocator);
                }
                return Err(error);
            }
        };
        let depth = depth.unwrap();
        let uniforms = uniforms.unwrap();
        let texture_set_layout = texture_set_layout.unwrap();
        let white_texture = white_texture.unwrap();
        let white_set = white_texture.set;
        let light = light.unwrap();
        let instance_buffer = instance_buffer.unwrap();

        let properties = unsafe {
            instance
                .instance
                .get_physical_device_properties(device.physical_device)
        };
        let device_name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let pipeline_formats = (swapchain.format, depth.format());
        Ok(Self {
            instance: ManuallyDrop::new(instance),
            device: ManuallyDrop::new(device),
            swapchain: ManuallyDrop::new(swapchain),
            pipeline: ManuallyDrop::new(pipeline),
            frame_context: ManuallyDrop::new(frame_context),
            uniforms: Some(uniforms),
            depth: Some(depth),
            texture_set_layout,
            white_texture: Some(white_texture),
            white_set,
            textures: Vec::new(),
            texture_sets: Vec::new(),
            meshes: Vec::new(),
            owner_id: handle::next_owner_id(),
            light: Some(light),
            directional_light: DirectionalLight::default(),
            point_lights: Vec::new(),
            instance_buffer: Some(instance_buffer),
            submit_state: SubmitState::NotSubmitted,
            allocator: ManuallyDrop::new(allocator),
            _window: window,
            pending_resize: None,
            text_renderer: None,
            pipeline_formats,
            text_vertices: Vec::new(),
            device_name,
            scene_draw_calls: 0,
            submitted_indices: 0,
            scene_instances: 0,
        })
    }

    /// Notify the renderer that the window surface resized to `new_extent` (physical pixels). The
    /// next `render_frame` recreates the swapchain before drawing; a zero dimension (minimized)
    /// pauses rendering until it's non-zero again.
    pub fn resize(&mut self, new_extent: (u32, u32)) {
        self.pending_resize = Some(new_extent);
    }

    /// Set the directional ("sun") light used for subsequent frames. Takes effect on the next
    /// `render_*` call, which uploads it into that frame's light uniform.
    /// Rejects invalid direction/intensity values (see `DirectionalLight`) with a validation error,
    /// preserving the previous light.
    pub fn set_directional_light(&mut self, light: DirectionalLight) -> Result<()> {
        self.directional_light.set_validated(light)
    }

    /// Set the point lights used for subsequent frames. Takes effect on the next `render_*` call,
    /// which uploads them into that frame's light uniform.
    /// Rejects a list longer than `MAX_POINT_LIGHTS` or containing an invalid light (see
    /// `PointLight`) with a validation error, preserving the previous list.
    pub fn set_point_lights(&mut self, lights: &[PointLight]) -> Result<()> {
        validate_point_lights(lights)?;
        self.point_lights = lights.to_vec();
        Ok(())
    }

    /// Reject any operation once the renderer has entered its terminal (poisoned) state — a fence wait
    /// could not confirm GPU completion, so the resource graph has been leaked rather than freed and
    /// nothing may be reused.
    fn ensure_not_terminal(&self) -> Result<()> {
        if self.submit_state.is_terminal() {
            return Err(VulkanError::Terminal(
                "renderer is in a terminal state after a failed GPU fence wait".into(),
            ));
        }
        Ok(())
    }

    /// Record a just-finished GPU operation's result: a terminal failure (from any entry point — a
    /// frame or an upload) poisons the renderer so every later call is rejected and `Drop` leaks the
    /// graph. Returns the result unchanged. The single place terminal-ness is entered.
    fn guard<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(error) = &result {
            self.submit_state = self.submit_state.poisoned_by(error);
        }
        result
    }

    /// Draw one frame of `mesh` under the given transform. `model` is the object's world-space
    /// transform (what the caller's simulation produces); `camera` carries the view/projection
    /// matrices plus world-space eye position — the renderer combines model/view/proj as
    /// `proj * view * model` in the vertex shader.
    pub fn render_frame(
        &mut self,
        mesh: mesh::MeshId,
        model: Mat4,
        camera: uniform::Camera,
    ) -> Result<()> {
        self.render_objects(
            &[Object {
                mesh,
                model,
                texture: None,
                specular: SpecularMaterial::default(),
            }],
            camera,
        )
    }

    /// Draw one copy of the uploaded mesh per object (with its own transform, texture, and specular
    /// material) under a shared camera. An empty slice clears and presents the frame without drawing
    /// geometry.
    pub fn render_objects(&mut self, objects: &[Object], camera: uniform::Camera) -> Result<()> {
        self.render_objects_with_text(objects, camera, &[])
    }

    /// Draw scene geometry, then screen-space text. Text is borrowed for this frame only.
    /// Invalid text/overflow/camera/material returns an error before acquiring or submitting a frame.
    pub fn render_objects_with_text(
        &mut self,
        objects: &[Object],
        camera: uniform::Camera,
        text: &[text::TextDraw<'_>],
    ) -> Result<()> {
        self.ensure_not_terminal()?;
        uniform::validate_camera(&camera)?;
        validate_object_count(objects.len())?;
        for object in objects {
            validate_specular(&object.specular)?;
            mesh::resolve_mesh(object.mesh, self.owner_id, &self.meshes)?;
            texture::resolve_texture_set(
                object.texture,
                self.owner_id,
                &self.texture_sets,
                self.white_set,
            )?;
        }
        if let Some((width, height)) = self.pending_resize {
            if width == 0 || height == 0 {
                // Minimized: can't build a swapchain. Keep the pending size and skip drawing.
                return Ok(());
            }
            self.wait_idle()?;
            self.swapchain.recreate(&self.device, (width, height))?;
            // The image count can change on recreate; rebuild the per-image render-finished
            // semaphores to match (device is idle from wait_idle above).
            let image_count = self.swapchain.images.len();
            self.frame_context
                .recreate_present_semaphores(&self.device, image_count)?;
            // The depth attachment is swapchain-extent-sized: recreate it to the new extent too
            // (device idle from wait_idle above). Keep the old attachment valid on allocation failure.
            let replacement = depth::DepthResources::new(
                &self.instance,
                &self.device,
                &mut self.allocator,
                self.swapchain.extent,
            )?;
            if let Some(old) = self.depth.take() {
                old.destroy(&self.device, &mut self.allocator);
            }
            self.depth = Some(replacement);
            // Dynamic viewport/scissor handles extent changes, but formats are pipeline state.
            if let Some(renderer) = self.text_renderer.as_mut() {
                renderer.ensure_formats(
                    &self.device,
                    self.swapchain.format,
                    self.depth.as_ref().unwrap().format(),
                )?;
            }
            if self.pipeline_formats
                != (self.swapchain.format, self.depth.as_ref().unwrap().format())
            {
                let pipeline = pipeline::GraphicsPipeline::new(
                    &self.device,
                    self.swapchain.format,
                    self.depth.as_ref().unwrap().format(),
                    &[
                        self.uniforms.as_ref().unwrap().descriptor_set_layout(),
                        self.texture_set_layout,
                        self.light.as_ref().unwrap().descriptor_set_layout(),
                    ],
                )?;
                let old = std::mem::replace(&mut self.pipeline, ManuallyDrop::new(pipeline));
                drop(ManuallyDrop::into_inner(old));
                self.pipeline_formats =
                    (self.swapchain.format, self.depth.as_ref().unwrap().format());
            }
            self.pending_resize = None;
        }

        text::layout_text(text, self.swapchain.extent, &mut self.text_vertices)?;
        if !self.text_vertices.is_empty() && self.text_renderer.is_none() {
            self.text_renderer = Some(text::TextRenderer::new(
                &self.device,
                &mut self.allocator,
                self.swapchain.format,
                self.depth.as_ref().unwrap().format(),
            )?);
        }
        let camera_uniforms = uniform::CameraUniforms::from(&camera);
        let outcome = self.frame_context.draw_frame(
            &self.device,
            &self.pipeline,
            &self.swapchain,
            self.depth.as_ref().unwrap(),
            self.uniforms.as_mut().unwrap(),
            &camera_uniforms,
            self.light.as_mut().unwrap(),
            &self.directional_light,
            &self.point_lights,
            self.instance_buffer.as_mut().unwrap(),
            objects,
            self.owner_id,
            &self.meshes,
            &self.texture_sets,
            self.white_set,
            self.text_renderer.as_mut(),
            &self.text_vertices,
        );
        let outcome = self.guard(outcome)?;
        if outcome.submitted() {
            // Re-walk the actual runs `emit_object_draws` grouped objects into (not a second,
            // parallel batching implementation) to derive all three counters consistently: one
            // draw call and index_count*instance_count indices per run, instance_count instances
            // per run. Every object handle was already proven to resolve by the validation pass
            // above, so this cannot fail. `vk::Buffer::null()` stands in for the instance buffer:
            // this pass only counts commands, it never records any of them.
            let mut draw_calls = 0usize;
            let mut indices = 0u64;
            let mut instances = 0usize;
            command::emit_object_draws(
                objects,
                self.owner_id,
                &self.meshes,
                &self.texture_sets,
                self.white_set,
                vk::Buffer::null(),
                |command| {
                    if let ObjectCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        ..
                    } = command
                    {
                        draw_calls += 1;
                        indices = indices.saturating_add(
                            u64::from(index_count).saturating_mul(u64::from(instance_count)),
                        );
                        instances = instances.saturating_add(instance_count as usize);
                    }
                },
            );
            self.scene_draw_calls = draw_calls;
            self.submitted_indices = indices;
            self.scene_instances = instances;
        }
        match outcome {
            command::FrameOutcome::Rendered => {}
            command::FrameOutcome::SwapchainOutOfDate { .. } => {
                // Recreate next frame; `recreate` re-queries the surface for the true size, so the
                // current extent is just a fallback hint.
                let extent = self.swapchain.extent;
                self.pending_resize = Some((extent.width, extent.height));
            }
        }
        Ok(())
    }

    /// Snapshot of the current surface and most recently submitted scene work.
    pub fn stats(&self) -> diagnostics::RendererStats<'_> {
        diagnostics::RendererStats {
            device_name: &self.device_name,
            extent: (self.swapchain.extent.width, self.swapchain.extent.height),
            color_format: self.swapchain.format,
            image_count: self.swapchain.images.len(),
            scene_draw_calls: self.scene_draw_calls,
            submitted_indices: self.submitted_indices,
            scene_instances: self.scene_instances,
            status: match self.pending_resize {
                Some((0, _)) | Some((_, 0)) => diagnostics::RenderStatus::Paused,
                Some(_) => diagnostics::RenderStatus::ResizePending,
                None => diagnostics::RenderStatus::Ready,
            },
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .map_err(VulkanError::from)?;
        }
        Ok(())
    }

    /// Upload an RGBA8 image (`rgba8.len() == width * height * 4`, both non-zero) into a new
    /// renderer-owned texture, returning its handle. The texture is freed when the renderer drops.
    /// Single-level (`Mipmaps::Disabled`) — equivalent to
    /// `create_texture_with_options(rgba8, width, height, TextureOptions::default())`.
    pub fn create_texture(
        &mut self,
        rgba8: &[u8],
        width: u32,
        height: u32,
    ) -> Result<texture::TextureId> {
        self.create_texture_with_options(rgba8, width, height, texture::TextureOptions::default())
    }

    /// Upload an RGBA8 image into a new renderer-owned texture with explicit `options` (e.g.
    /// `Mipmaps::Generate` to generate a full mip chain). The texture is freed when the renderer
    /// drops. See spec 014.
    pub fn create_texture_with_options(
        &mut self,
        rgba8: &[u8],
        width: u32,
        height: u32,
        options: texture::TextureOptions,
    ) -> Result<texture::TextureId> {
        self.ensure_not_terminal()?;
        let stored = StoredTexture::new(
            &self.device,
            &mut self.allocator,
            &self.instance,
            self.texture_set_layout,
            rgba8,
            width,
            height,
            options,
        );
        let stored = self.guard(stored)?;
        let id = texture::TextureId::new(self.owner_id, self.textures.len());
        self.texture_sets.push(stored.set);
        self.textures.push(stored);
        Ok(id)
    }

    /// Upload `vertices`/`indices` into a new renderer-owned mesh, returning its handle. The mesh is
    /// freed when the renderer drops.
    ///
    /// ```no_run
    /// use m2s2_rust_vulkan::{Camera, Index, Object, Result, SpecularMaterial, Vertex, VulkanRenderer};
    /// use m2s2_rust_vulkan::math::Mat4;
    /// fn draw(renderer: &mut VulkanRenderer, vertices: &[Vertex], indices: &[Index], camera: Camera) -> Result<()> {
    ///     let mesh = renderer.create_mesh(vertices, indices)?;
    ///     let objects = [Object { mesh, model: Mat4::IDENTITY, texture: None,
    ///         specular: SpecularMaterial::default() }];
    ///     renderer.render_objects(&objects, camera)?;
    ///     renderer.render_frame(mesh, Mat4::IDENTITY, camera)
    /// }
    /// ```
    pub fn create_mesh(
        &mut self,
        vertices: &[buffer::Vertex],
        indices: &[buffer::Index],
    ) -> Result<mesh::MeshId> {
        self.ensure_not_terminal()?;
        let stored = mesh::upload_mesh(&self.device, &mut self.allocator, vertices, indices);
        let stored = self.guard(stored)?;
        let id = mesh::MeshId::new(self.owner_id, self.meshes.len());
        self.meshes.push(stored);
        Ok(id)
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // Terminal state: a fence wait could not confirm GPU completion, so freeing anything risks
        // releasing memory the GPU may still touch. Deliberately leak the entire resource graph
        // (every owner, incl. the retained window) rather than free or abort. See spec 012.
        if self.submit_state.is_terminal() {
            log::error!(
                "renderer in terminal state: leaking the GPU resource graph to avoid freeing \
                 possibly in-use memory"
            );
            std::mem::forget(self.uniforms.take());
            std::mem::forget(self.light.take());
            std::mem::forget(self.instance_buffer.take());
            std::mem::forget(self.depth.take());
            std::mem::forget(self.white_texture.take());
            std::mem::forget(self.text_renderer.take());
            std::mem::forget(std::mem::take(&mut self.textures));
            std::mem::forget(std::mem::take(&mut self.meshes));
            std::mem::forget(self._window.take());
            // The `ManuallyDrop` owners (instance/device/swapchain/pipeline/frame_context/allocator)
            // are intentionally never dropped here, so they leak too.
            return;
        }

        log::info!("Destroying Vulkan renderer");
        let _ = self.wait_idle();

        // Explicit order matters here: each of these must be destroyed before the
        // VkDevice/VkInstance it was created from. `ManuallyDrop::drop` is `unsafe` because
        // it's on us to guarantee it's called exactly once per field, which the compiler
        // can't check for us the way it can for ordinary owned values.
        // Reverse of construction order: frame_context and pipeline must be destroyed before the
        // device/instance they were built from.
        unsafe {
            ManuallyDrop::drop(&mut self.frame_context);
            if let Some(text) = self.text_renderer.take() {
                text.destroy(&self.device, &mut self.allocator);
            }
            // Pipeline before uniforms: its pipeline layout references the descriptor-set layout
            // that `uniforms` owns and destroys.
            ManuallyDrop::drop(&mut self.pipeline);
            // Free everything allocator-backed (uniforms' buffers, depth image, mesh/texture
            // buffers) before dropping the allocator -- every allocation must outlive neither the
            // allocator nor the device it was built from. Each owner is moved out of its `Option`
            // and consumed exactly once.
            if let Some(uniforms) = self.uniforms.take() {
                uniforms.destroy(&self.device, &mut self.allocator);
            }
            if let Some(light) = self.light.take() {
                light.destroy(&self.device, &mut self.allocator);
            }
            if let Some(instance_buffer) = self.instance_buffer.take() {
                instance_buffer.destroy(&self.device, &mut self.allocator);
            }
            if let Some(depth) = self.depth.take() {
                depth.destroy(&self.device, &mut self.allocator);
            }
            for mesh in std::mem::take(&mut self.meshes) {
                mesh.destroy(&self.device, &mut self.allocator);
            }
            for texture in std::mem::take(&mut self.textures) {
                texture.destroy(&self.device, &mut self.allocator);
            }
            if let Some(white_texture) = self.white_texture.take() {
                white_texture.destroy(&self.device, &mut self.allocator);
            }
            self.device
                .device
                .destroy_descriptor_set_layout(self.texture_set_layout, None);
            ManuallyDrop::drop(&mut self.allocator);
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}

/// Rejects an `objects` slice longer than `MAX_INSTANCES_PER_FRAME` before any per-object
/// validation, swapchain recreation, or GPU work. Pure, so it's unit-testable without a GPU (same
/// shape as `instance_buffer::validate_instance_count`, which this mirrors) — but runs much
/// earlier, in `render_objects_with_text`'s up-front validation pass, so an oversized frame fails
/// with a normal `ValidationError` instead of reaching `InstanceBuffer::write` deep inside
/// `draw_frame`, where a failure is treated as terminal.
fn validate_object_count(len: usize) -> Result<()> {
    if len > MAX_INSTANCES_PER_FRAME {
        return Err(VulkanError::ValidationError(format!(
            "frame has {len} objects, exceeds MAX_INSTANCES_PER_FRAME ({MAX_INSTANCES_PER_FRAME})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_object_count_accepts_up_to_max() {
        assert!(validate_object_count(0).is_ok());
        assert!(validate_object_count(MAX_INSTANCES_PER_FRAME).is_ok());
    }

    #[test]
    fn validate_object_count_rejects_over_max() {
        assert!(matches!(
            validate_object_count(MAX_INSTANCES_PER_FRAME + 1),
            Err(VulkanError::ValidationError(_))
        ));
    }
}
