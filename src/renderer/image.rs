// Shared single-channel sampled image upload, initially used by the bitmap font atlas.
use super::mip::{self, BlitStep, MipPolicy};
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

/// The number of mip levels a `MipPolicy` allocates: one for `OneLevel` (and for a degenerate
/// `Generate { level_count: 1 }`, e.g. a 1x1 texture), or the computed `level_count` otherwise.
fn levels_for_policy(policy: MipPolicy) -> u32 {
    match policy {
        MipPolicy::OneLevel => 1,
        MipPolicy::Generate { level_count } => level_count,
    }
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
            vk::SamplerMipmapMode::NEAREST,
            MipPolicy::OneLevel,
        )
    }

    pub fn new_rgba8_srgb(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        pixels: &[u8],
        policy: MipPolicy,
    ) -> Result<Self> {
        Self::new(
            device,
            allocator,
            extent,
            pixels,
            vk::Format::R8G8B8A8_SRGB,
            4,
            vk::Filter::LINEAR,
            vk::SamplerMipmapMode::LINEAR,
            policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        pixels: &[u8],
        format: vk::Format,
        bytes_per_pixel: u32,
        filter: vk::Filter,
        mipmap_mode: vk::SamplerMipmapMode,
        policy: MipPolicy,
    ) -> Result<Self> {
        validate_pixels(extent, pixels.len(), bytes_per_pixel)?;
        let levels = levels_for_policy(policy);
        let mut staging = Buffer::new_host_visible_mapped(
            device,
            allocator,
            pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        staging.write_mapped(pixels)?;
        let mut resource = Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            allocation: None,
        };
        let dev = &device.device;
        let result = (|| {
            let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
            if levels > 1 {
                usage |= vk::ImageUsageFlags::TRANSFER_SRC;
            }
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(levels)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
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
                // (1) Transition every allocated level from UNDEFINED to TRANSFER_DST_OPTIMAL —
                // including levels above 0, which are written only by the blits below, not by this
                // copy.
                let all_levels = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(levels)
                    .layer_count(1);
                let to_dst = vk::ImageMemoryBarrier::default()
                    .image(resource.image)
                    .subresource_range(all_levels)
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
                    &[to_dst],
                );

                // (2) Copy the caller's base pixels from staging into mip 0 only.
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

                // (3) Blit each level from its predecessor, in order.
                for step in mip::plan_blits(extent.width, extent.height, levels) {
                    record_blit_step(dev, cmd, resource.image, &step);
                }

                // (4) Transition the final level (the base level when there were no blits) from
                // TRANSFER_DST_OPTIMAL to SHADER_READ_ONLY_OPTIMAL.
                let last_level = levels - 1;
                let last_level_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(last_level)
                    .level_count(1)
                    .layer_count(1);
                let to_shader_read = vk::ImageMemoryBarrier::default()
                    .image(resource.image)
                    .subresource_range(last_level_range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
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
                    &[to_shader_read],
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
                                .level_count(levels)
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
                        .mipmap_mode(mipmap_mode)
                        .min_lod(0.0)
                        .max_lod((levels - 1) as f32)
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

    pub fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe {
            device.device.destroy_sampler(self.sampler, None);
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation {
            let _ = allocator.free(allocation);
        }
    }
}

/// Records one step of the mip chain's generation: `src_level` goes TRANSFER_DST_OPTIMAL ->
/// TRANSFER_SRC_OPTIMAL (ordering its prior write before the blit's read), the blit itself
/// (`vk::Filter::LINEAR`, per the spec's sampler/filtering choice), then `src_level` goes
/// TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL (ordering the blit's read before any later
/// fragment-shader sampling). `dst_level` is left in TRANSFER_DST_OPTIMAL — either consumed as the
/// next step's `src_level`, or finalized to SHADER_READ_ONLY_OPTIMAL by the caller's last-level
/// barrier.
fn record_blit_step(dev: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image, step: &BlitStep) {
    unsafe {
        let src_level_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(step.src_level)
            .level_count(1)
            .layer_count(1);
        let to_blit_src = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(src_level_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_blit_src],
        );

        let (src_w, src_h) = step.src_extent;
        let (dst_w, dst_h) = step.dst_extent;
        let blit = vk::ImageBlit::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(step.src_level)
                    .layer_count(1),
            )
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: src_w as i32,
                    y: src_h as i32,
                    z: 1,
                },
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(step.dst_level)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: dst_w as i32,
                    y: dst_h as i32,
                    z: 1,
                },
            ]);
        dev.cmd_blit_image(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        let to_shader_read = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(src_level_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_shader_read],
        );
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

    #[test]
    fn one_level_policy_allocates_exactly_one_level_and_plans_no_blits() {
        assert_eq!(levels_for_policy(MipPolicy::OneLevel), 1);
        assert_eq!(levels_for_policy(MipPolicy::Generate { level_count: 1 }), 1);
        assert_eq!(levels_for_policy(MipPolicy::Generate { level_count: 5 }), 5);
    }
}
