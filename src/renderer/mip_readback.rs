//! Test-only GPU readback of one mip level of a texture image, for spec 014's acceptance tests
//! (verifying generated mip content is real GPU output, not a stub). Mirrors `upload.rs`'s
//! one-time-submission shape: a transient command buffer, submitted and waited on via
//! `upload::submit_once`, copying into a host-visible staging `Buffer` that is mapped and read
//! back afterward.
//!
//! A completed texture upload (`image::SampledImage::new`) leaves every mip level in
//! `SHADER_READ_ONLY_OPTIMAL` — this helper transitions the requested `level` out to
//! `TRANSFER_SRC_OPTIMAL` for the copy, then back to `SHADER_READ_ONLY_OPTIMAL` afterward, so the
//! image remains usable by its owning `HeadlessRenderer` exactly as before the read (the image is
//! not destroyed here; normal teardown happens via the renderer's own `Drop`).

use super::{allocator::Allocator, buffer::Buffer, device::VulkanDevice, upload};
use crate::error::{Result, VulkanError};
use ash::vk;

/// Copy mip `level` (whose extent is `width`x`height`) of `image` back to the CPU as tightly
/// packed RGBA8 bytes. `image` must be a texture created via the production uploader (so every
/// level is currently `SHADER_READ_ONLY_OPTIMAL`); `level` must be a valid level of that image.
pub(crate) fn read_back_mip_level(
    device: &VulkanDevice,
    allocator: &mut Allocator,
    image: vk::Image,
    level: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let byte_len = u64::from(width) * u64::from(height) * 4;
    let staging = Buffer::new_host_visible_mapped(
        device,
        allocator,
        byte_len,
        vk::BufferUsageFlags::TRANSFER_DST,
    )?;

    let level_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(level)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let submit_result = upload::submit_once(device, |dev, cmd| unsafe {
        // (1) `level` goes SHADER_READ_ONLY_OPTIMAL -> TRANSFER_SRC_OPTIMAL, ordering any prior
        // shader read before the copy's read.
        let to_transfer_src = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(level_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
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
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(level)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        dev.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging.buffer,
            &[copy],
        );

        // (3) `level` goes back TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL, ordering the
        // copy's read before any later fragment-shader sampling -- the image stays usable
        // afterward, it is not being torn down here.
        let back_to_shader_read = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(level_range)
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
