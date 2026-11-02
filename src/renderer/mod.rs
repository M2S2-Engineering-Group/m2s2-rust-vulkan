pub mod allocator;
pub mod buffer;
pub mod command;
pub mod device;
pub mod instance;
pub mod pipeline;
pub mod swapchain;

use crate::error::{Result, VulkanError};
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
    pub allocator: ManuallyDrop<allocator::Allocator>,
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
        let vertex_buffer = buffer::Buffer::new_device_local(
            &device,
            &mut allocator,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let index_buffer = buffer::Buffer::new_device_local(
            &device,
            &mut allocator,
            indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        let pipeline = pipeline::GraphicsPipeline::new(&device, swapchain.format)?;
        // FrameContext::new also takes &instance to build the VK_KHR_dynamic_rendering loader
        // (see command.rs) -- a small tweak from spec 002's `new(&device)`.
        let frame_context = command::FrameContext::new(&instance, &device)?;

        Ok(Self {
            instance: ManuallyDrop::new(instance),
            device: ManuallyDrop::new(device),
            swapchain: ManuallyDrop::new(swapchain),
            pipeline: ManuallyDrop::new(pipeline),
            frame_context: ManuallyDrop::new(frame_context),
            vertex_buffer,
            index_buffer,
            allocator: ManuallyDrop::new(allocator),
        })
    }

    pub fn render_frame(&mut self) -> Result<()> {
        self.frame_context.draw_frame(
            &self.device,
            &self.pipeline,
            &self.swapchain,
            &self.vertex_buffer,
            &self.index_buffer,
        )
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
            ManuallyDrop::drop(&mut self.pipeline);
            // Free the buffers back to the allocator, then drop the allocator -- every allocation
            // must outlive neither the allocator nor the device it was built from.
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
