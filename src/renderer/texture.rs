//! Caller-facing texture handles and the renderer-owned texture resource.
use super::{allocator::Allocator, device::VulkanDevice, image::SampledImage};
use crate::{
    error::{Result, VulkanError},
    math::Mat4,
};
use ash::vk;

/// Opaque handle to a renderer-owned texture (an index into the renderer's texture store).
/// `Copy`; valid for the lifetime of the renderer that produced it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TextureId(pub(super) usize);

/// One drawn copy of the uploaded mesh. `texture: None` samples the renderer's 1x1 white default,
/// so an untextured object reproduces the pre-texture look (`white * vertexColor == vertexColor`).
pub struct Object {
    pub model: Mat4,
    pub texture: Option<TextureId>,
}

const TEXTURE_BINDING: u32 = 0;

pub(super) fn create_set_layout(device: &VulkanDevice) -> Result<vk::DescriptorSetLayout> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(TEXTURE_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    unsafe {
        device.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(VulkanError::from)
}

pub(crate) struct StoredTexture {
    image: SampledImage,
    pool: vk::DescriptorPool,
    pub(super) set: vk::DescriptorSet,
}

impl StoredTexture {
    pub(super) fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        layout: vk::DescriptorSetLayout,
        rgba8: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // SampledImage::new_rgba8_srgb validates and rolls back its own partial resources on error.
        let mut image =
            SampledImage::new_rgba8_srgb(device, allocator, vk::Extent2D { width, height }, rgba8)?;
        let dev = &device.device;
        // Allocate the descriptor pool/set after the image. On any error here the image is already
        // owned, so destroy it before returning so nothing leaks.
        let result = (|| {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)];
            let pool = unsafe {
                dev.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
            }
            .map_err(VulkanError::from)?;
            let set = unsafe {
                dev.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[layout]),
                )
            }
            .map_err(VulkanError::from)?[0];
            let info = [vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .sampler(image.sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            unsafe {
                dev.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(TEXTURE_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info)],
                    &[],
                )
            };
            Ok((pool, set))
        })();
        match result {
            Ok((pool, set)) => Ok(Self { image, pool, set }),
            Err(error) => {
                image.destroy(device, allocator);
                Err(error)
            }
        }
    }

    pub(super) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe { device.device.destroy_descriptor_pool(self.pool, None) };
        self.image.destroy(device, allocator);
    }
}

/// The descriptor set bound at set 1 for an object: the stored texture's set, or `white` when the
/// object has no texture or references an out-of-range handle (a stale/foreign id falls back to
/// white rather than indexing out of bounds).
pub(super) fn resolve_texture_set(
    texture: Option<TextureId>,
    texture_sets: &[vk::DescriptorSet],
    white: vk::DescriptorSet,
) -> vk::DescriptorSet {
    match texture {
        None => white,
        Some(TextureId(index)) => texture_sets.get(index).copied().unwrap_or(white),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle; // brings `from_raw` into scope

    fn set(raw: u64) -> vk::DescriptorSet {
        vk::DescriptorSet::from_raw(raw)
    }

    #[test]
    fn resolve_none_and_out_of_range_fall_back_to_white() {
        let white = set(0x10);
        let sets = [set(0xa0), set(0xb0)];
        assert_eq!(resolve_texture_set(None, &sets, white), white);
        assert_eq!(
            resolve_texture_set(Some(TextureId(0)), &sets, white),
            sets[0]
        );
        assert_eq!(
            resolve_texture_set(Some(TextureId(1)), &sets, white),
            sets[1]
        );
        // out of range -> white, not a panic
        assert_eq!(resolve_texture_set(Some(TextureId(2)), &sets, white), white);
        assert_eq!(resolve_texture_set(Some(TextureId(0)), &[], white), white);
    }
}
