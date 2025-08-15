pub mod instance;
pub mod device;
pub mod swapchain;
pub mod pipeline;
pub mod buffer;
pub mod command;

use ash::vk;
use crate::error::{Result, VulkanError};
use crate::window::Window;

pub struct VulkanRenderer {
    pub instance: instance::VulkanInstance,
    pub device: device::VulkanDevice,
    pub swapchain: swapchain::VulkanSwapchain,
}

impl VulkanRenderer {
    pub fn new(window: &Window) -> Result<Self> {
        log::info!("Initializing Vulkan renderer");
        
        let instance = instance::VulkanInstance::new()?;
        let device = device::VulkanDevice::new(&instance, window)?;
        let swapchain = swapchain::VulkanSwapchain::new(&instance, &device, window)?;

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
