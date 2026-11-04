pub mod allocator;
pub mod buffer;
mod cleanup;
pub mod command;
pub mod depth;
pub mod device;
pub mod diagnostics;
mod image;
pub mod instance;
pub mod pipeline;
pub mod swapchain;
pub mod text;
pub mod uniform;
mod upload;

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::mem::ManuallyDrop;

pub struct VulkanRenderer {
    // `ManuallyDrop` and torn down explicitly in `Drop`, in swapchain -> device -> instance
    // order, instead of relying on Rust's default field-drop order. That default drops struct
    // fields top-to-bottom (declaration order) -- the opposite of what's needed here, since
    // Vulkan requires every object be destroyed before the VkDevice/VkInstance it was created
    // from. See the `Drop` impl below.
    pub instance: ManuallyDrop<instance::VulkanInstance>,
    pub device: ManuallyDrop<device::VulkanDevice>,
    pub swapchain: ManuallyDrop<swapchain::VulkanSwapchain>,
    // Constructed after swapchain; torn down before it in Drop (reverse construction order), so
    // the full destroy chain is frame_context -> pipeline -> swapchain -> device -> instance.
    pub pipeline: ManuallyDrop<pipeline::GraphicsPipeline>,
    pub frame_context: ManuallyDrop<command::FrameContext>,
    // Caller-supplied geometry uploaded into GPU buffers, plus the allocator backing them. The
    // buffers are NOT ManuallyDrop: freeing a gpu-allocator allocation needs `&mut Allocator`, so
    // they're freed via explicit `destroy` in Drop, before the allocator (and the allocator before
    // the device).
    pub vertex_buffer: buffer::Buffer,
    pub index_buffer: buffer::Buffer,
    // Per-frame camera uniforms + descriptor sets, and the depth attachment. Like the buffers (and for
    // the same reason — freeing their allocations needs `&mut Allocator`), these are NOT
    // `ManuallyDrop`; they're torn down via explicit `destroy` in `Drop`, before the allocator.
    // `depth` is also recreated on resize (it's swapchain-extent-sized); `uniforms` is not.
    pub(crate) uniforms: uniform::Uniforms,
    pub(crate) depth: depth::DepthResources,
    pub allocator: ManuallyDrop<allocator::Allocator>,
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
    /// `window_extent` is the surface size in physical pixels, e.g. a winit window's `inner_size()`.
    /// `vertices`/`indices` are the caller-supplied geometry to draw — the library defines no
    /// meshes of its own. They're uploaded into GPU buffers this renderer owns.
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
        vertices: &[buffer::Vertex],
        indices: &[buffer::Index],
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

            let pipeline = pipeline::GraphicsPipeline::new(
                &device,
                swapchain.format,
                depth.as_ref().unwrap().format(),
                uniforms.as_ref().unwrap().descriptor_set_layout(),
            )?;
            // FrameContext::new also takes &instance to build the VK_KHR_dynamic_rendering loader
            // (see command.rs) -- a small tweak from spec 002's `new(&device)`.
            let frame_context =
                command::FrameContext::new(&instance, &device, swapchain.images.len())?;

            Ok((pipeline, frame_context))
        })();
        let (pipeline, frame_context) = match resources {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(resource) = uniforms.as_mut() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = depth.as_mut() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = index_buffer.as_mut() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = vertex_buffer.as_mut() {
                    resource.destroy(&device, &mut allocator);
                }
                return Err(error);
            }
        };
        let vertex_buffer = vertex_buffer.unwrap();
        let index_buffer = index_buffer.unwrap();
        let depth = depth.unwrap();
        let uniforms = uniforms.unwrap();

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
            vertex_buffer,
            index_buffer,
            uniforms,
            depth,
            allocator: ManuallyDrop::new(allocator),
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

    /// Draw one frame of the caller-supplied geometry under the given transform. `model` is the
    /// object's world-space transform (what the caller's simulation produces), `view`/`proj` the
    /// camera — the renderer combines them as `proj * view * model` in the vertex shader.
    pub fn render_frame(&mut self, model: Mat4, view: Mat4, proj: Mat4) -> Result<()> {
        self.render_objects(std::slice::from_ref(&model), view, proj)
    }

    /// Draw one copy of the uploaded mesh per model transform with a shared camera.
    /// An empty slice clears and presents the frame without drawing geometry.
    pub fn render_objects(&mut self, models: &[Mat4], view: Mat4, proj: Mat4) -> Result<()> {
        self.render_objects_with_text(models, view, proj, &[])
    }

    /// Draw scene geometry, then screen-space text. Text is borrowed for this frame only.
    /// Invalid text/overflow returns an error before acquiring or submitting a frame.
    pub fn render_objects_with_text(
        &mut self,
        models: &[Mat4],
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
            self.depth.destroy(&self.device, &mut self.allocator);
            self.depth = replacement;
            // Dynamic viewport/scissor handles extent changes, but formats are pipeline state.
            if let Some(renderer) = self.text_renderer.as_mut() {
                renderer.ensure_formats(
                    &self.device,
                    self.swapchain.format,
                    self.depth.format(),
                )?;
            }
            if self.pipeline_formats != (self.swapchain.format, self.depth.format()) {
                let pipeline = pipeline::GraphicsPipeline::new(
                    &self.device,
                    self.swapchain.format,
                    self.depth.format(),
                    self.uniforms.descriptor_set_layout(),
                )?;
                let old = std::mem::replace(&mut self.pipeline, ManuallyDrop::new(pipeline));
                drop(ManuallyDrop::into_inner(old));
                self.pipeline_formats = (self.swapchain.format, self.depth.format());
            }
            self.pending_resize = None;
        }

        text::layout_text(text, self.swapchain.extent, &mut self.text_vertices)?;
        if !self.text_vertices.is_empty() && self.text_renderer.is_none() {
            self.text_renderer = Some(text::TextRenderer::new(
                &self.device,
                &mut self.allocator,
                self.swapchain.format,
                self.depth.format(),
            )?);
        }
        let camera = uniform::CameraUniforms { view, proj };
        let outcome = self.frame_context.draw_frame(
            &self.device,
            &self.pipeline,
            &self.swapchain,
            &self.depth,
            &self.uniforms,
            &self.vertex_buffer,
            &self.index_buffer,
            &camera,
            models,
            self.text_renderer.as_ref(),
            &self.text_vertices,
        )?;
        if outcome.submitted() {
            self.scene_draw_calls = models.len();
            self.submitted_indices = models.len() as u64 * u64::from(self.index_buffer.len());
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
            if let Some(mut text) = self.text_renderer.take() {
                text.destroy(&self.device, &mut self.allocator);
            }
            // Pipeline before uniforms: its pipeline layout references the descriptor-set layout
            // that `uniforms` owns and destroys.
            ManuallyDrop::drop(&mut self.pipeline);
            // Free everything allocator-backed (uniforms' buffers, depth image, vertex/index
            // buffers) before dropping the allocator -- every allocation must outlive neither the
            // allocator nor the device it was built from.
            self.uniforms.destroy(&self.device, &mut self.allocator);
            self.depth.destroy(&self.device, &mut self.allocator);
            self.vertex_buffer
                .destroy(&self.device, &mut self.allocator);
            self.index_buffer.destroy(&self.device, &mut self.allocator);
            ManuallyDrop::drop(&mut self.allocator);
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}
