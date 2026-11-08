//! Caller-facing texture handles and the renderer-owned texture resource.
use super::{allocator::Allocator, device::VulkanDevice, image::SampledImage};
use crate::{
    error::{Result, VulkanError},
    math::{Mat4, Vec3},
};
use ash::vk;

/// Opaque handle to a renderer-owned texture (an index into the renderer's texture store).
/// `Copy`; valid for the lifetime of the renderer that produced it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TextureId(pub(super) usize);

/// Per-object Blinn-Phong specular parameters (spec 013). `color` tints the highlight; `shininess`
/// controls its tightness (higher = smaller, sharper highlight).
#[derive(Copy, Clone, Debug)]
pub struct SpecularMaterial {
    pub color: Vec3,
    pub shininess: f32,
}

impl Default for SpecularMaterial {
    /// No highlight (`color` black) and a moderate shininess exponent — reproduces the pre-013 look
    /// until a caller opts in with a non-black color.
    fn default() -> Self {
        Self {
            color: Vec3::new(0.0, 0.0, 0.0),
            shininess: 32.0,
        }
    }
}

/// Validate a `SpecularMaterial` before any GPU work: `color` components must be finite and
/// nonnegative; `shininess` must be finite and at least 1.0 (the minimum meaningful Blinn-Phong
/// exponent).
pub(crate) fn validate_specular(material: &SpecularMaterial) -> Result<()> {
    if !(0..3).all(|i| material.color[i].is_finite() && material.color[i] >= 0.0) {
        return Err(VulkanError::ValidationError(
            "specular color must be finite and nonnegative".into(),
        ));
    }
    if !(material.shininess.is_finite() && material.shininess >= 1.0) {
        return Err(VulkanError::ValidationError(
            "specular shininess must be finite and at least 1.0".into(),
        ));
    }
    Ok(())
}

/// One drawn copy of the uploaded mesh. `texture: None` samples the renderer's 1x1 white default,
/// so an untextured object reproduces the pre-texture look (`white * vertexColor == vertexColor`).
pub struct Object {
    pub model: Mat4,
    pub texture: Option<TextureId>,
    pub specular: SpecularMaterial,
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
        let image =
            SampledImage::new_rgba8_srgb(device, allocator, vk::Extent2D { width, height }, rgba8)?;
        let dev = &device.device;
        // Pool is created outside the allocate/write closure below so its handle is available to
        // the error arm: if allocation or the descriptor write fails, the pool must still be
        // destroyed (it would otherwise leak — nothing else owns it at that point) before the
        // image is torn down.
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)];
        let pool = match unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                image.destroy(device, allocator);
                return Err(VulkanError::from(error));
            }
        };
        let result = (|| {
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
            Ok(set)
        })();
        match result {
            Ok(set) => Ok(Self { image, pool, set }),
            Err(error) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                image.destroy(device, allocator);
                Err(error)
            }
        }
    }

    pub(super) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
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
    use crate::math::Vec3;
    use ash::vk::Handle; // brings `from_raw` into scope

    #[test]
    fn specular_material_default_matches_pre_013_look() {
        let m = SpecularMaterial::default();
        assert_eq!((m.color, m.shininess), (Vec3::new(0.0, 0.0, 0.0), 32.0));
    }

    #[test]
    fn validate_specular_rejects_invalid_color_and_shininess() {
        let base = SpecularMaterial::default();
        assert!(validate_specular(&base).is_ok());
        for bad in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for i in 0..3 {
                let mut m = base;
                m.color[i] = bad;
                assert!(validate_specular(&m).is_err());
            }
        }
        for bad_shininess in [0.0, 0.5, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_specular(&SpecularMaterial {
                color: base.color,
                shininess: bad_shininess
            })
            .is_err());
        }
        assert!(validate_specular(&SpecularMaterial {
            color: base.color,
            shininess: 1.0
        })
        .is_ok());
    }

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
