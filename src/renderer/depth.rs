// Depth attachment for the main render pass. Swapchain-sized, recreated on resize. See
// specs/006-uniforms-depth-camera.md.

use crate::error::{Result, VulkanError};
use crate::renderer::allocator::Allocator;
use crate::renderer::device::VulkanDevice;
use crate::renderer::instance::VulkanInstance;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

// Depth formats in preference order: plain 32-bit float depth first (no stencil, widely supported),
// then the combined depth/stencil formats as fallbacks.
const DEPTH_FORMAT_CANDIDATES: [vk::Format; 3] = [
    vk::Format::D32_SFLOAT,
    vk::Format::D32_SFLOAT_S8_UINT,
    vk::Format::D24_UNORM_S8_UINT,
];

/// The depth image + view backing the main pass's depth test. Device-local, sized to the swapchain
/// extent, so it is destroyed and recreated whenever the swapchain is (resize). Not `Drop`: freeing
/// its allocation needs `&mut Allocator` (same reason as `Buffer`), so call `destroy` explicitly.
pub(crate) struct DepthResources {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    format: vk::Format,
}

impl DepthResources {
    pub(crate) fn new(
        instance: &VulkanInstance,
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let candidates: Vec<(vk::Format, vk::FormatFeatureFlags)> = DEPTH_FORMAT_CANDIDATES
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
        let format = select_depth_format(&candidates).ok_or_else(|| {
            VulkanError::InitializationError("no supported depth-stencil attachment format".into())
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
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe {
            device
                .device
                .create_image(&image_info, None)
                .map_err(VulkanError::from)?
        };

        let requirements = unsafe { device.device.get_image_memory_requirements(image) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "depth",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false, // OPTIMAL-tiled image
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(VulkanError::from)?;
        }

        // View is depth-aspect only — the combined-format fallbacks carry a stencil aspect this pass
        // never uses.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
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

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            format,
        })
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }

    /// Destroy the view + image (via `device`) and free the allocation (via `allocator`). Must run
    /// before the `Allocator` is dropped.
    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe {
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Picks the depth format from `candidates`, which the caller supplies in priority order paired with
/// each format's queried `optimal_tiling_features`. Returns the first format whose features advertise
/// `DEPTH_STENCIL_ATTACHMENT` (i.e. it's usable as a depth attachment with OPTIMAL tiling), or `None`
/// if none qualify. Pure given the queried properties, so it's testable without a device.
fn select_depth_format(candidates: &[(vk::Format, vk::FormatFeatureFlags)]) -> Option<vk::Format> {
    candidates
        .iter()
        .find(|(_, features)| features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT))
        .map(|&(format, _)| format)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEPTH_ATTACHMENT: vk::FormatFeatureFlags =
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;

    #[test]
    fn prefers_first_candidate_that_supports_depth_attachment() {
        let candidates = [
            (vk::Format::D32_SFLOAT, DEPTH_ATTACHMENT),
            (vk::Format::D24_UNORM_S8_UINT, DEPTH_ATTACHMENT),
        ];
        assert_eq!(
            select_depth_format(&candidates),
            Some(vk::Format::D32_SFLOAT)
        );
    }

    #[test]
    fn falls_back_when_first_candidate_unsupported() {
        let candidates = [
            (vk::Format::D32_SFLOAT, vk::FormatFeatureFlags::empty()),
            (vk::Format::D24_UNORM_S8_UINT, DEPTH_ATTACHMENT),
        ];
        assert_eq!(
            select_depth_format(&candidates),
            Some(vk::Format::D24_UNORM_S8_UINT)
        );
    }

    #[test]
    fn returns_none_when_no_candidate_supports_depth_attachment() {
        let candidates = [
            (vk::Format::D32_SFLOAT, vk::FormatFeatureFlags::empty()),
            (
                vk::Format::D24_UNORM_S8_UINT,
                vk::FormatFeatureFlags::empty(),
            ),
        ];
        assert_eq!(select_depth_format(&candidates), None);
    }
}
