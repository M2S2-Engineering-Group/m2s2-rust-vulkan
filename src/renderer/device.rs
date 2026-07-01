use ash::{vk, Device};
use crate::error::{Result, VulkanError};
use crate::renderer::instance::VulkanInstance;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct VulkanDevice {
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub present_queue_family: u32,
}

impl VulkanDevice {
    pub fn new(
        instance: &VulkanInstance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Self> {
        let physical_device = Self::pick_physical_device(instance)?;
        let queue_families =
            Self::find_queue_families(instance, physical_device, display_handle, window_handle)?;
        
        let device_extensions = vec![
            ash::khr::swapchain::NAME.as_ptr(),
        ];

        let queue_priorities = [1.0];
        let mut queue_create_infos = vec![];

        // Graphics queue
        let graphics_queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_families.graphics_family)
            .queue_priorities(&queue_priorities);
        queue_create_infos.push(graphics_queue_create_info);

        // Present queue (if different from graphics)
        if queue_families.present_family != queue_families.graphics_family {
            let present_queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_families.present_family)
                .queue_priorities(&queue_priorities);
            queue_create_infos.push(present_queue_create_info);
        }

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features);

        let device = unsafe {
            instance.instance.create_device(physical_device, &device_create_info, None)
                .map_err(VulkanError::from)?
        };

        let graphics_queue = unsafe {
            device.get_device_queue(queue_families.graphics_family, 0)
        };

        let present_queue = unsafe {
            device.get_device_queue(queue_families.present_family, 0)
        };

        Ok(Self {
            physical_device,
            device,
            graphics_queue,
            present_queue,
            graphics_queue_family: queue_families.graphics_family,
            present_queue_family: queue_families.present_family,
        })
    }

    fn pick_physical_device(instance: &VulkanInstance) -> Result<vk::PhysicalDevice> {
        let physical_devices = unsafe {
            instance.instance.enumerate_physical_devices()
                .map_err(VulkanError::from)?
        };

        if physical_devices.is_empty() {
            return Err(VulkanError::InitializationError(
                "No Vulkan-compatible devices found".to_string()
            ));
        }

        // For now, just pick the first device
        // TODO: Add device suitability scoring
        Ok(physical_devices[0])
    }

    fn find_queue_families(
        instance: &VulkanInstance,
        physical_device: vk::PhysicalDevice,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<QueueFamilyIndices> {
        let queue_families = unsafe {
            instance.instance.get_physical_device_queue_family_properties(physical_device)
        };

        let mut graphics_family = None;
        let mut present_family = None;

        // Create a surface to check present support
        let surface = Self::create_surface(instance, display_handle, window_handle)?;
        let surface_loader = ash::khr::surface::Instance::new(&instance.entry, &instance.instance);

        for (index, queue_family) in queue_families.iter().enumerate() {
            let index = index as u32;

            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(index);
            }

            let present_support = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index,
                    surface,
                ).map_err(VulkanError::from)?
            };

            if present_support {
                present_family = Some(index);
            }

            if graphics_family.is_some() && present_family.is_some() {
                break;
            }
        }

        // Clean up surface
        unsafe {
            surface_loader.destroy_surface(surface, None);
        }

        match (graphics_family, present_family) {
            (Some(graphics), Some(present)) => Ok(QueueFamilyIndices {
                graphics_family: graphics,
                present_family: present,
            }),
            _ => Err(VulkanError::InitializationError(
                "Could not find suitable queue families".to_string()
            )),
        }
    }

    fn create_surface(
        instance: &VulkanInstance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<vk::SurfaceKHR> {
        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.instance,
                display_handle,
                window_handle,
                None,
            ).map_err(VulkanError::from)?
        };

        Ok(surface)
    }
}

struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
