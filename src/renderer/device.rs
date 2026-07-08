use crate::error::{Result, VulkanError};
use crate::renderer::instance::VulkanInstance;
use ash::{vk, Device};
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
        // Only used to probe present support/swapchain adequacy while picking a device;
        // destroyed before returning. The real, long-lived surface is created by VulkanSwapchain.
        let surface = instance.create_surface(display_handle, window_handle)?;
        let surface_loader = ash::khr::surface::Instance::new(&instance.entry, &instance.instance);

        let (physical_device, queue_families) =
            Self::pick_physical_device(instance, &surface_loader, surface)?;

        let device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];

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
            instance
                .instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(VulkanError::from)?
        };

        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_family, 0) };

        let present_queue = unsafe { device.get_device_queue(queue_families.present_family, 0) };

        unsafe {
            surface_loader.destroy_surface(surface, None);
        }

        Ok(Self {
            physical_device,
            device,
            graphics_queue,
            present_queue,
            graphics_queue_family: queue_families.graphics_family,
            present_queue_family: queue_families.present_family,
        })
    }

    /// Picks the best-scoring physical device that has a graphics queue, a present queue for
    /// `surface`, the `VK_KHR_swapchain` extension, and at least one supported surface format
    /// and present mode. Devices missing any of those are skipped rather than causing a hard
    /// error, so one bad device doesn't block picking a suitable one.
    fn pick_physical_device(
        instance: &VulkanInstance,
        surface_loader: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, QueueFamilyIndices)> {
        let physical_devices = unsafe {
            instance
                .instance
                .enumerate_physical_devices()
                .map_err(VulkanError::from)?
        };

        if physical_devices.is_empty() {
            return Err(VulkanError::InitializationError(
                "No Vulkan-compatible devices found".to_string(),
            ));
        }

        let mut best: Option<(u32, vk::PhysicalDevice, QueueFamilyIndices)> = None;

        for physical_device in physical_devices {
            let Some(queue_families) =
                Self::find_queue_families(instance, surface_loader, physical_device, surface)?
            else {
                continue;
            };

            if !Self::supports_required_extensions(instance, physical_device)? {
                continue;
            }

            if !Self::has_adequate_swapchain_support(surface_loader, physical_device, surface)? {
                continue;
            }

            let device_type = unsafe {
                instance
                    .instance
                    .get_physical_device_properties(physical_device)
            }
            .device_type;
            let score = Self::score_device_type(device_type);

            if best
                .as_ref()
                .is_none_or(|(best_score, ..)| score > *best_score)
            {
                best = Some((score, physical_device, queue_families));
            }
        }

        best.map(|(_, physical_device, queue_families)| (physical_device, queue_families))
            .ok_or_else(|| {
                VulkanError::InitializationError(
                    "No Vulkan device supports the required queues, extensions, and swapchain for this surface"
                        .to_string(),
                )
            })
    }

    /// Higher is more preferred. Discrete GPUs are strongly preferred over integrated/virtual/CPU
    /// implementations, matching the standard suitability-scoring pattern for Vulkan device selection.
    fn score_device_type(device_type: vk::PhysicalDeviceType) -> u32 {
        match device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 250,
            vk::PhysicalDeviceType::CPU => 100,
            _ => 0,
        }
    }

    fn supports_required_extensions(
        instance: &VulkanInstance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<bool> {
        let available_extensions = unsafe {
            instance
                .instance
                .enumerate_device_extension_properties(physical_device)
                .map_err(VulkanError::from)?
        };

        Ok(available_extensions
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(ash::khr::swapchain::NAME)))
    }

    fn has_adequate_swapchain_support(
        surface_loader: &ash::khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<bool> {
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(VulkanError::from)?
        };
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .map_err(VulkanError::from)?
        };

        Ok(!formats.is_empty() && !present_modes.is_empty())
    }

    /// Returns `None` (rather than an error) when `physical_device` lacks a graphics queue or a
    /// present-capable queue for `surface`, so the caller can skip it during device selection.
    fn find_queue_families(
        instance: &VulkanInstance,
        surface_loader: &ash::khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Option<QueueFamilyIndices>> {
        let queue_families = unsafe {
            instance
                .instance
                .get_physical_device_queue_family_properties(physical_device)
        };

        let mut graphics_family = None;
        let mut present_family = None;

        for (index, queue_family) in queue_families.iter().enumerate() {
            let index = index as u32;

            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(index);
            }

            let present_support = unsafe {
                surface_loader
                    .get_physical_device_surface_support(physical_device, index, surface)
                    .map_err(VulkanError::from)?
            };

            if present_support {
                present_family = Some(index);
            }

            if graphics_family.is_some() && present_family.is_some() {
                break;
            }
        }

        Ok(match (graphics_family, present_family) {
            (Some(graphics_family), Some(present_family)) => Some(QueueFamilyIndices {
                graphics_family,
                present_family,
            }),
            _ => None,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_gpu_scores_highest() {
        assert!(
            VulkanDevice::score_device_type(vk::PhysicalDeviceType::DISCRETE_GPU)
                > VulkanDevice::score_device_type(vk::PhysicalDeviceType::INTEGRATED_GPU)
        );
    }

    #[test]
    fn integrated_gpu_beats_virtual_and_cpu() {
        assert!(
            VulkanDevice::score_device_type(vk::PhysicalDeviceType::INTEGRATED_GPU)
                > VulkanDevice::score_device_type(vk::PhysicalDeviceType::VIRTUAL_GPU)
        );
        assert!(
            VulkanDevice::score_device_type(vk::PhysicalDeviceType::VIRTUAL_GPU)
                > VulkanDevice::score_device_type(vk::PhysicalDeviceType::CPU)
        );
    }

    #[test]
    fn unknown_device_type_scores_zero() {
        assert_eq!(
            VulkanDevice::score_device_type(vk::PhysicalDeviceType::OTHER),
            0
        );
    }
}
