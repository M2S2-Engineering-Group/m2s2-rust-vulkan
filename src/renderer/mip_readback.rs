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

use super::{
    allocator::Allocator,
    device::VulkanDevice,
    image_readback::{read_back_image, ImageReadback},
};
use crate::error::Result;
use ash::vk;

pub(crate) fn read_back_mip_level(
    device: &VulkanDevice,
    allocator: &mut Allocator,
    image: vk::Image,
    level: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    read_back_image(
        device,
        allocator,
        image,
        ImageReadback {
            level,
            width,
            height,
            bytes_per_texel: 4,
            aspect: vk::ImageAspectFlags::COLOR,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
    )
}
