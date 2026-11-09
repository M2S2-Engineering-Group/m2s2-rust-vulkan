// Offscreen color render target for headless (no-surface) rendering in GPU tests. Stands in for a
// swapchain image: no surface, no present — just a plain color `VkImage` the one-shot recorder
// (Task 3) renders into and reads back from. See specs/019-headless-rendering-and-golden-image-tests.md.

use crate::error::{Result, VulkanError};
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::Buffer;
use crate::renderer::device::VulkanDevice;
use crate::renderer::instance::VulkanInstance;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// Color formats in preference order: plain RGBA8 first, then the swapchain-typical BGRA8 as a
/// fallback.
pub(crate) const OFFSCREEN_COLOR_FORMAT_CANDIDATES: [vk::Format; 2] =
    [vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM];

/// A plain color image + view standing in for a swapchain image in headless tests, plus a
/// host-visible readback buffer for pulling rendered pixels back to the CPU. Device-local, sized to
/// `extent` at construction (not recreated on resize — headless tests have no resize). Not `Drop`:
/// freeing its allocation (and the readback buffer's) needs `&mut Allocator`, so call `destroy`
/// explicitly.
pub(crate) struct OffscreenTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    readback: Buffer,
    extent: vk::Extent2D,
    format: vk::Format,
    has_rendered: bool,
}

impl OffscreenTarget {
    pub(crate) fn new(
        device: &VulkanDevice,
        instance: &VulkanInstance,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let candidates: Vec<(vk::Format, vk::FormatFeatureFlags)> =
            OFFSCREEN_COLOR_FORMAT_CANDIDATES
                .iter()
                .map(|&format| {
                    let props = unsafe {
                        instance
                            .instance
                            .get_physical_device_format_properties(device.physical_device, format)
                    };
                    (format, props.optimal_tiling_features)
                })
                .collect();
        let format = select_color_format(&candidates).ok_or_else(|| {
            VulkanError::InitializationError(
                "no supported offscreen color-attachment format".into(),
            )
        })?;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe {
            device
                .device
                .create_image(&image_info, None)
                .map_err(VulkanError::from)?
        };

        let requirements = unsafe { device.device.get_image_memory_requirements(image) };
        let allocation_result = allocator.allocate(&AllocationCreateDesc {
            name: "offscreen-color",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false, // OPTIMAL-tiled image
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation_result {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.device.destroy_image(image, None) };
                return Err(error);
            }
        };
        let result = (|| {
            unsafe {
                device
                    .device
                    .bind_image_memory(image, allocation.memory(), allocation.offset())
                    .map_err(VulkanError::from)?;
            }

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            let view = unsafe {
                device
                    .device
                    .create_image_view(&view_info, None)
                    .map_err(VulkanError::from)?
            };

            Ok(view)
        })();
        let view = match result {
            Ok(view) => view,
            Err(error) => {
                unsafe { device.device.destroy_image(image, None) };
                let _ = allocator.free(allocation);
                return Err(error);
            }
        };

        let readback_result = Buffer::new_host_visible_mapped(
            device,
            allocator,
            extent.width as u64 * extent.height as u64 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        let readback = match readback_result {
            Ok(readback) => readback,
            Err(error) => {
                unsafe {
                    device.device.destroy_image_view(view, None);
                    device.device.destroy_image(image, None);
                }
                let _ = allocator.free(allocation);
                return Err(error);
            }
        };

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            readback,
            extent,
            format,
            has_rendered: false,
        })
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }

    pub(crate) fn has_rendered(&self) -> bool {
        self.has_rendered
    }

    pub(crate) fn mark_rendered(&mut self) {
        self.has_rendered = true;
    }

    /// Destroy the view + image (via `device`) and free the image allocation and the readback
    /// buffer (via `allocator`). Must run before the `Allocator` is dropped.
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe {
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation {
            let _ = allocator.free(allocation);
        }
        self.readback.destroy(device, allocator);
    }
}

/// Picks the color format from `candidates`, which the caller supplies in priority order paired
/// with each format's queried `optimal_tiling_features`. Returns the first format whose features
/// advertise `COLOR_ATTACHMENT` (i.e. it's usable as a color attachment with OPTIMAL tiling), or
/// `None` if none qualify. Pure given the queried properties, so it's testable without a device.
fn select_color_format(candidates: &[(vk::Format, vk::FormatFeatureFlags)]) -> Option<vk::Format> {
    candidates
        .iter()
        .find(|(_, features)| features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT))
        .map(|&(format, _)| format)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLOR_ATTACHMENT: vk::FormatFeatureFlags = vk::FormatFeatureFlags::COLOR_ATTACHMENT;

    #[test]
    fn prefers_first_candidate_that_supports_color_attachment() {
        let candidates = [
            (vk::Format::R8G8B8A8_UNORM, COLOR_ATTACHMENT),
            (vk::Format::B8G8R8A8_UNORM, COLOR_ATTACHMENT),
        ];
        assert_eq!(
            select_color_format(&candidates),
            Some(vk::Format::R8G8B8A8_UNORM)
        );
    }

    #[test]
    fn falls_back_when_first_candidate_unsupported() {
        let candidates = [
            (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::empty()),
            (vk::Format::B8G8R8A8_UNORM, COLOR_ATTACHMENT),
        ];
        assert_eq!(
            select_color_format(&candidates),
            Some(vk::Format::B8G8R8A8_UNORM)
        );
    }

    #[test]
    fn returns_none_when_no_candidate_supports_color_attachment() {
        let candidates = [
            (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::empty()),
            (vk::Format::B8G8R8A8_UNORM, vk::FormatFeatureFlags::empty()),
        ];
        assert_eq!(select_color_format(&candidates), None);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn offscreen_target_constructs_tracks_render_state_and_tears_down() {
        let instance = super::super::instance::VulkanInstance::new_headless().expect("instance");
        let device = super::super::device::VulkanDevice::new_headless(&instance).expect("device");
        let mut allocator =
            super::super::allocator::Allocator::new(&instance, &device).expect("allocator");
        let mut target = OffscreenTarget::new(
            &device,
            &instance,
            &mut allocator,
            vk::Extent2D {
                width: 4,
                height: 4,
            },
        )
        .expect("offscreen target");
        assert!(!target.has_rendered());
        assert_eq!(
            target.extent(),
            vk::Extent2D {
                width: 4,
                height: 4,
            }
        );
        assert!(matches!(
            target.format(),
            vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
        ));
        target.mark_rendered();
        assert!(target.has_rendered());
        target.destroy(&device, &mut allocator);
    }
}
