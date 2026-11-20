//! Test-only sampled-image readback shared by texture mip and shadow depth acceptance tests.
use super::{allocator::Allocator, buffer::Buffer, device::VulkanDevice, upload};
use crate::error::{Result, VulkanError};
use ash::vk;

pub(super) struct ImageReadback {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_texel: u32,
    pub aspect: vk::ImageAspectFlags,
    pub layout: vk::ImageLayout,
}

/// The image must be in `request.layout` and prior writes must be complete. Restores that layout.
pub(super) fn read_back_image(
    device: &VulkanDevice,
    allocator: &mut Allocator,
    image: vk::Image,
    request: ImageReadback,
) -> Result<Vec<u8>> {
    let byte_len =
        u64::from(request.width) * u64::from(request.height) * u64::from(request.bytes_per_texel);
    let staging = Buffer::new_host_visible_mapped(
        device,
        allocator,
        byte_len,
        vk::BufferUsageFlags::TRANSFER_DST,
    )?;

    let level_range = vk::ImageSubresourceRange::default()
        .aspect_mask(request.aspect)
        .base_mip_level(request.level)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let submit_result = upload::submit_once(device, |dev, cmd| unsafe {
        // (1) The requested subresource goes from its sampled layout -> TRANSFER_SRC_OPTIMAL, ordering any prior
        // shader read before the copy's read.
        let to_transfer_src = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(level_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(request.layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_src],
        );

        // (2) Copy the level's pixels into the staging buffer.
        let copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(request.aspect)
                    .mip_level(request.level)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: request.width,
                height: request.height,
                depth: 1,
            });
        dev.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging.buffer,
            &[copy],
        );

        // (3) The requested subresource returns to its sampled layout, ordering the
        // copy's read before any later fragment-shader sampling -- the image stays usable
        // afterward, it is not being torn down here.
        let back_to_shader_read = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(level_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(request.layout)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[back_to_shader_read],
        );
    });

    match submit_result {
        Ok(()) => {
            let bytes = staging.mapped_bytes().ok_or_else(|| {
                VulkanError::ValidationError("readback buffer is not mapped".into())
            })?[..byte_len as usize]
                .to_vec();
            staging.destroy(device, allocator);
            Ok(bytes)
        }
        Err(error) => {
            // Mirrors `upload::submit_once`'s own caution: if it returned `Err`, completion wasn't
            // confirmed, so the GPU may still own `staging` -- leak it rather than risk freeing a
            // buffer still in flight. Test-only helper; the process exits shortly after a failure
            // here regardless.
            std::mem::forget(staging);
            Err(error)
        }
    }
}
