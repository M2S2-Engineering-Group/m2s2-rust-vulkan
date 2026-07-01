pub mod instance;
pub mod device;
pub mod swapchain;
pub mod pipeline;
pub mod buffer;
pub mod command;

use crate::error::{Result, VulkanError};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct VulkanRenderer {
    pub instance: instance::VulkanInstance,
    pub device: device::VulkanDevice,
    pub swapchain: swapchain::VulkanSwapchain,
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

        Ok(Self {
            instance,
            device,
            swapchain,
        })
    }

    pub fn render_frame(&mut self) -> Result<()> {
        // Basic render loop - to be implemented
        log::debug!("Rendering frame");
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.device.device.device_wait_idle()
                .map_err(VulkanError::from)?;
        }
        Ok(())
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        log::info!("Destroying Vulkan renderer");
        let _ = self.wait_idle();
    }
}
