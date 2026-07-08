use crate::error::{Result, VulkanError};
use crate::renderer::{device::VulkanDevice, instance::VulkanInstance};
use ash::{khr::swapchain, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct VulkanSwapchain {
    pub swapchain_loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl VulkanSwapchain {
    pub fn new(
        instance: &VulkanInstance,
        device: &VulkanDevice,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
    ) -> Result<Self> {
        let surface = instance.create_surface(display_handle, window_handle)?;
        let surface_loader = ash::khr::surface::Instance::new(&instance.entry, &instance.instance);

        let swapchain_support =
            Self::query_swapchain_support(&surface_loader, device.physical_device, surface)?;

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(&swapchain_support.capabilities, window_extent);

        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count > 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain_loader = swapchain::Device::new(&instance.instance, &device.device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .map_err(VulkanError::from)?
        };

        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(VulkanError::from)?
        };

        let image_views = Self::create_image_views(&device.device, &images, surface_format.format)?;

        Ok(Self {
            swapchain_loader,
            swapchain,
            images,
            image_views,
            format: surface_format.format,
            extent,
            surface_loader,
            surface,
        })
    }

    fn query_swapchain_support(
        surface_loader: &ash::khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapchainSupportDetails> {
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(VulkanError::from)?
        };

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

        Ok(SwapchainSupportDetails {
            capabilities,
            formats,
            present_modes,
        })
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        for format in available_formats {
            if format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }
        available_formats[0]
    }

    fn choose_swap_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        for &mode in available_present_modes {
            if mode == vk::PresentModeKHR::MAILBOX {
                return mode;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window_extent: (u32, u32),
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let (width, height) = window_extent;
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>> {
        let mut image_views = Vec::with_capacity(images.len());

        for &image in images {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_view = unsafe {
                device
                    .create_image_view(&create_info, None)
                    .map_err(VulkanError::from)?
            };

            image_views.push(image_view);
        }

        Ok(image_views)
    }
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        // Note: image_views need to be destroyed by the device that owns them
        // This will be handled by the renderer cleanup
        unsafe {
            // The swapchain must be destroyed before the surface it was created from
            // (VUID-vkDestroySurfaceKHR-surface-01266).
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space,
        }
    }

    #[test]
    fn prefers_srgb_when_available() {
        let formats = [
            format(
                vk::Format::R8G8B8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            format(vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
        ];
        let chosen = VulkanSwapchain::choose_swap_surface_format(&formats);
        assert_eq!(chosen.format, vk::Format::B8G8R8A8_SRGB);
        assert_eq!(chosen.color_space, vk::ColorSpaceKHR::SRGB_NONLINEAR);
    }

    #[test]
    fn falls_back_to_first_format_when_srgb_unavailable() {
        let formats = [format(
            vk::Format::R8G8B8A8_UNORM,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        )];
        let chosen = VulkanSwapchain::choose_swap_surface_format(&formats);
        assert_eq!(chosen.format, vk::Format::R8G8B8A8_UNORM);
    }

    #[test]
    fn prefers_mailbox_present_mode_when_available() {
        let modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        assert_eq!(
            VulkanSwapchain::choose_swap_present_mode(&modes),
            vk::PresentModeKHR::MAILBOX
        );
    }

    #[test]
    fn falls_back_to_fifo_present_mode() {
        let modes = [vk::PresentModeKHR::IMMEDIATE];
        assert_eq!(
            VulkanSwapchain::choose_swap_present_mode(&modes),
            vk::PresentModeKHR::FIFO
        );
    }

    fn capabilities(
        current_extent: vk::Extent2D,
        min_extent: vk::Extent2D,
        max_extent: vk::Extent2D,
    ) -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            current_extent,
            min_image_extent: min_extent,
            max_image_extent: max_extent,
            ..Default::default()
        }
    }

    #[test]
    fn uses_current_extent_when_fixed_by_surface() {
        let caps = capabilities(
            vk::Extent2D {
                width: 640,
                height: 480,
            },
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Extent2D {
                width: 4096,
                height: 4096,
            },
        );
        let extent = VulkanSwapchain::choose_swap_extent(&caps, (1920, 1080));
        assert_eq!(extent.width, 640);
        assert_eq!(extent.height, 480);
    }

    #[test]
    fn clamps_window_extent_when_surface_extent_is_variable() {
        let caps = capabilities(
            vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            vk::Extent2D {
                width: 100,
                height: 100,
            },
            vk::Extent2D {
                width: 800,
                height: 600,
            },
        );
        let extent = VulkanSwapchain::choose_swap_extent(&caps, (50, 2000));
        assert_eq!(extent.width, 100);
        assert_eq!(extent.height, 600);
    }
}
