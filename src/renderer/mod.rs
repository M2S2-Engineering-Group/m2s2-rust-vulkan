pub mod buffer;
pub mod command;
pub mod device;
pub mod instance;
pub mod pipeline;
pub mod swapchain;

use crate::error::{Result, VulkanError};
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
    // TODO(spec 002): add, in this order (construction order -- see spec 002's "Destroy order"
    // section for why the *destroy* order below needs to be the reverse of this):
    //     pub pipeline: ManuallyDrop<pipeline::GraphicsPipeline>,
    //     pub frame_context: ManuallyDrop<command::FrameContext>,
    // Same pattern as the three fields above -- you've already done this once for the
    // instance/device/swapchain fix, this is the same move two fields deeper.
}

impl VulkanRenderer {
    /// `window_extent` is the surface size in physical pixels, e.g. a winit window's `inner_size()`.
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
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

        // TODO(spec 002): after swapchain is built above, construct the pipeline and frame
        // context too, before this Ok(Self { ... }):
        //     let pipeline = pipeline::GraphicsPipeline::new(&device, swapchain.format)?;
        //     let frame_context = command::FrameContext::new(&device)?;

        Ok(Self {
            instance: ManuallyDrop::new(instance),
            device: ManuallyDrop::new(device),
            swapchain: ManuallyDrop::new(swapchain),
            // TODO(spec 002): pipeline: ManuallyDrop::new(pipeline),
            // TODO(spec 002): frame_context: ManuallyDrop::new(frame_context),
        })
    }

    pub fn render_frame(&mut self) -> Result<()> {
        // TODO(spec 002): replace this whole body with:
        //     self.frame_context.draw_frame(&self.device, &self.pipeline, &self.swapchain)
        log::debug!("Rendering frame");
        Ok(())
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
        // TODO(spec 002): add `ManuallyDrop::drop(&mut self.frame_context);` and
        // `ManuallyDrop::drop(&mut self.pipeline);` to this block, ABOVE the swapchain line --
        // both must be destroyed before device (and don't touch swapchain themselves, so their
        // position relative to the swapchain line doesn't matter, only relative to device/instance).
        unsafe {
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}
