// Shared single-channel sampled image upload, initially used by the bitmap font atlas.
use super::{allocator::Allocator, buffer::Buffer, device::VulkanDevice};
use crate::error::{Result, VulkanError};
use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};

pub(super) struct SampledImage {
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    image: vk::Image,
    allocation: Option<Allocation>,
}

impl SampledImage {
    pub fn new_r8(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        pixels: &[u8],
    ) -> Result<Self> {
        Self::new(
            device,
            allocator,
            extent,
            pixels,
            vk::Format::R8_UNORM,
            1,
            vk::Filter::NEAREST,
        )
    }

    pub fn new_rgba8_srgb(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        pixels: &[u8],
    ) -> Result<Self> {
        Self::new(
            device,
            allocator,
            extent,
            pixels,
            vk::Format::R8G8B8A8_SRGB,
            4,
            vk::Filter::LINEAR,
        )
    }

    fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        pixels: &[u8],
        format: vk::Format,
        bytes_per_pixel: u32,
        filter: vk::Filter,
    ) -> Result<Self> {
        validate_pixels(extent, pixels.len(), bytes_per_pixel)?;
        let mut staging = Buffer::new_host_visible_mapped(
            device,
            allocator,
            pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        staging.write_mapped(pixels);
        let mut resource = Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            allocation: None,
        };
        let dev = &device.device;
        let result = (|| {
            let info = vk::ImageCreateInfo::default()
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
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            resource.image = unsafe { dev.create_image(&info, None) }?;
            resource.allocation = Some(allocator.allocate(&AllocationCreateDesc {
                name: "sampled image",
                requirements: unsafe { dev.get_image_memory_requirements(resource.image) },
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?);
            let allocation = resource.allocation.as_ref().unwrap();
            unsafe {
                dev.bind_image_memory(resource.image, allocation.memory(), allocation.offset())
            }?;
            super::upload::submit_once(device, |dev, cmd| unsafe {
                let range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1);
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(resource.image)
                    .subresource_range(range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
                let region = vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    });
                dev.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    resource.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                let barrier = barrier
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            })?;
            resource.view = unsafe {
                dev.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(resource.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            }?;
            resource.sampler = unsafe {
                dev.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(filter)
                        .min_filter(filter)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
            }?;
            Ok(())
        })();
        staging.destroy(device, allocator);
        if let Err(error) = result {
            resource.destroy(device, allocator);
            return Err(error);
        }
        Ok(resource)
    }

    pub fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe {
            device.device.destroy_sampler(self.sampler, None);
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

fn validate_pixels(extent: vk::Extent2D, len: usize, bytes_per_pixel: u32) -> Result<()> {
    if extent.width == 0
        || extent.height == 0
        || u64::from(extent.width) * u64::from(extent.height) * u64::from(bytes_per_pixel)
            != len as u64
    {
        return Err(VulkanError::ValidationError(
            "sampled image extent must match its pixel bytes".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_pixel_byte_count_by_bpp() {
        // 1 bpp (font path)
        assert!(validate_pixels(
            vk::Extent2D {
                width: 2,
                height: 3
            },
            6,
            1
        )
        .is_ok());
        assert!(validate_pixels(
            vk::Extent2D {
                width: 2,
                height: 3
            },
            5,
            1
        )
        .is_err());
        // 4 bpp (RGBA textures)
        assert!(validate_pixels(
            vk::Extent2D {
                width: 2,
                height: 3
            },
            24,
            4
        )
        .is_ok());
        assert!(validate_pixels(
            vk::Extent2D {
                width: 2,
                height: 3
            },
            23,
            4
        )
        .is_err());
        // zero extent
        assert!(validate_pixels(
            vk::Extent2D {
                width: 0,
                height: 3
            },
            0,
            4
        )
        .is_err());
    }
}
