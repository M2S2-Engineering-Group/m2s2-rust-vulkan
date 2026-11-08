pub mod allocator;
pub mod buffer;
mod cleanup;
pub mod command;
pub mod depth;
pub mod device;
pub mod diagnostics;
mod image;
pub mod instance;
pub mod light;
pub(crate) mod per_frame_uniform;
pub mod pipeline;
pub mod swapchain;
pub mod text;
pub mod texture;
pub mod uniform;
mod upload;

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use ash::vk;
use light::{DirectionalLight, Light};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;

/// A retained window/display owner the renderer keeps alive for its whole lifetime, so the safe
/// `with_window` constructor upholds the surface's window-lifetime contract by construction. Blanket-
/// implemented for anything exposing both handle traits (e.g. a winit `Window`).
pub(crate) trait WindowOwner: HasWindowHandle + HasDisplayHandle {}
impl<T: HasWindowHandle + HasDisplayHandle> WindowOwner for T {}
use std::mem::ManuallyDrop;
use texture::{Object, StoredTexture};

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
///     let _ = m2s2_rust_vulkan::VulkanRenderer::new(d, w, (1, 1), &[], &[]);
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
    // Caller-supplied geometry uploaded into GPU buffers, plus the allocator backing them. The
    // buffers are NOT ManuallyDrop: freeing a gpu-allocator allocation needs `&mut Allocator`, so
    // they're freed via explicit `destroy` in Drop, before the allocator (and the allocator before
    // the device).
    // Owner resources held in `Option` so `Drop`/resize can move them into their ownership-consuming
    // `destroy(self, ..)` (a second destroy is then impossible). Always `Some` outside teardown.
    pub(crate) vertex_buffer: Option<buffer::Buffer>,
    pub(crate) index_buffer: Option<buffer::Buffer>,
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
    // Per-frame light uniform (descriptor set 2) and the current directional light it uploads each
    // frame. Like `uniforms`, freed via explicit `destroy` in `Drop` before the allocator.
    pub(crate) light: Option<Light>,
    pub(crate) directional_light: DirectionalLight,
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
}

impl VulkanRenderer {
    /// Safe constructor: the renderer retains shared ownership of `window`, so the surface's
    /// window/display connection stays valid for the renderer's whole lifetime — including teardown —
    /// with no caller `unsafe`. `window` provides both handles (e.g. an `Arc<winit::window::Window>`);
    /// keep your own clone for the event loop. `window_extent` is the surface size in physical pixels.
    pub fn with_window<W: HasWindowHandle + HasDisplayHandle + 'static>(
        window: Arc<W>,
        window_extent: (u32, u32),
        vertices: &[buffer::Vertex],
        indices: &[buffer::Index],
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
            vertices,
            indices,
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
    /// `window_extent` is the surface size in physical pixels; `vertices`/`indices` are the
    /// caller-supplied geometry, uploaded into GPU buffers this renderer owns.
    pub unsafe fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
        vertices: &[buffer::Vertex],
        indices: &[buffer::Index],
    ) -> Result<Self> {
        Self::init(
            display_handle,
            window_handle,
            window_extent,
            vertices,
            indices,
            None,
        )
    }

    /// Shared construction. `window` is the retained owner for the safe path (`None` for raw handles).
    fn init(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
        vertices: &[buffer::Vertex],
        indices: &[buffer::Index],
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

        // Allocator, then the two device-local buffers uploaded from the caller's geometry.
        let mut allocator = allocator::Allocator::new(&instance, &device)?;
        let mut vertex_buffer = None;
        let mut index_buffer = None;
        let mut depth = None;
        let mut uniforms = None;
        let mut texture_set_layout = None;
        let mut white_texture = None;
        let mut light = None;
        let resources = (|| {
            vertex_buffer = Some(buffer::Buffer::new_device_local(
                &device,
                &mut allocator,
                vertices,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?);
            index_buffer = Some(buffer::Buffer::new_device_local(
                &device,
                &mut allocator,
                indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?);

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
                layout,
                &[255, 255, 255, 255],
                1,
                1,
            )?);

            light = Some(Light::new(&device, &mut allocator)?);

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
                if let Some(resource) = index_buffer.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = vertex_buffer.take() {
                    resource.destroy(&device, &mut allocator);
                }
                return Err(error);
            }
        };
        let vertex_buffer = vertex_buffer.unwrap();
        let index_buffer = index_buffer.unwrap();
        let depth = depth.unwrap();
        let uniforms = uniforms.unwrap();
        let texture_set_layout = texture_set_layout.unwrap();
        let white_texture = white_texture.unwrap();
        let white_set = white_texture.set;
        let light = light.unwrap();

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
            vertex_buffer: Some(vertex_buffer),
            index_buffer: Some(index_buffer),
            uniforms: Some(uniforms),
            depth: Some(depth),
            texture_set_layout,
            white_texture: Some(white_texture),
            white_set,
            textures: Vec::new(),
            texture_sets: Vec::new(),
            light: Some(light),
            directional_light: DirectionalLight::default(),
            allocator: ManuallyDrop::new(allocator),
            _window: window,
            pending_resize: None,
            text_renderer: None,
            pipeline_formats,
            text_vertices: Vec::new(),
            device_name,
            scene_draw_calls: 0,
            submitted_indices: 0,
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

    /// Draw one frame of the caller-supplied geometry under the given transform. `model` is the
    /// object's world-space transform (what the caller's simulation produces), `view`/`proj` the
    /// camera — the renderer combines them as `proj * view * model` in the vertex shader.
    pub fn render_frame(&mut self, model: Mat4, view: Mat4, proj: Mat4) -> Result<()> {
        self.render_objects(
            &[Object {
                model,
                texture: None,
            }],
            view,
            proj,
        )
    }

    /// Draw one copy of the uploaded mesh per object (with its own transform and texture) under a
    /// shared camera. An empty slice clears and presents the frame without drawing geometry.
    pub fn render_objects(&mut self, objects: &[Object], view: Mat4, proj: Mat4) -> Result<()> {
        self.render_objects_with_text(objects, view, proj, &[])
    }

    /// Draw scene geometry, then screen-space text. Text is borrowed for this frame only.
    /// Invalid text/overflow returns an error before acquiring or submitting a frame.
    pub fn render_objects_with_text(
        &mut self,
        objects: &[Object],
        view: Mat4,
        proj: Mat4,
        text: &[text::TextDraw<'_>],
    ) -> Result<()> {
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
        let camera = uniform::CameraUniforms { view, proj };
        let outcome = self.frame_context.draw_frame(
            &self.device,
            &self.pipeline,
            &self.swapchain,
            self.depth.as_ref().unwrap(),
            self.uniforms.as_mut().unwrap(),
            self.vertex_buffer.as_ref().unwrap(),
            self.index_buffer.as_ref().unwrap(),
            &camera,
            self.light.as_mut().unwrap(),
            &self.directional_light,
            objects,
            &self.texture_sets,
            self.white_set,
            self.text_renderer.as_mut(),
            &self.text_vertices,
        )?;
        if outcome.submitted() {
            self.scene_draw_calls = objects.len();
            self.submitted_indices =
                objects.len() as u64 * u64::from(self.index_buffer.as_ref().unwrap().len());
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
    pub fn create_texture(
        &mut self,
        rgba8: &[u8],
        width: u32,
        height: u32,
    ) -> Result<texture::TextureId> {
        let stored = StoredTexture::new(
            &self.device,
            &mut self.allocator,
            self.texture_set_layout,
            rgba8,
            width,
            height,
        )?;
        let id = texture::TextureId(self.textures.len());
        self.texture_sets.push(stored.set);
        self.textures.push(stored);
        Ok(id)
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
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
            // Free everything allocator-backed (uniforms' buffers, depth image, vertex/index
            // buffers) before dropping the allocator -- every allocation must outlive neither the
            // allocator nor the device it was built from. Each owner is moved out of its `Option`
            // and consumed exactly once.
            if let Some(uniforms) = self.uniforms.take() {
                uniforms.destroy(&self.device, &mut self.allocator);
            }
            if let Some(light) = self.light.take() {
                light.destroy(&self.device, &mut self.allocator);
            }
            if let Some(depth) = self.depth.take() {
                depth.destroy(&self.device, &mut self.allocator);
            }
            if let Some(vertex_buffer) = self.vertex_buffer.take() {
                vertex_buffer.destroy(&self.device, &mut self.allocator);
            }
            if let Some(index_buffer) = self.index_buffer.take() {
                index_buffer.destroy(&self.device, &mut self.allocator);
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
