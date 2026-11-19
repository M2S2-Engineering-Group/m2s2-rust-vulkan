//! Caller-facing texture handles and the renderer-owned texture resource.
use super::{
    allocator::Allocator, device::VulkanDevice, handle, image::SampledImage,
    instance::VulkanInstance, mip,
};
use crate::{
    error::{Result, VulkanError},
    math::Vec3,
};
use ash::vk;

/// Mipmap generation policy for a texture created via `create_texture_with_options`. `Disabled`
/// preserves the existing single-level behavior; `Generate` requests a full mip chain, failing
/// `create_texture_with_options` (before any GPU submission) if the selected device can't support
/// the multi-level blit path for the requested extent — see spec 014's "Capability checks and
/// explicit failure policy".
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mipmaps {
    Disabled,
    Generate,
}

/// Options for `VulkanRenderer::create_texture_with_options`/
/// `HeadlessRenderer::create_texture_with_options`.
#[derive(Copy, Clone, Debug)]
pub struct TextureOptions {
    pub mipmaps: Mipmaps,
}

impl Default for TextureOptions {
    /// `Mipmaps::Disabled` — preserves `create_texture`'s existing single-level behavior.
    fn default() -> Self {
        Self {
            mipmaps: Mipmaps::Disabled,
        }
    }
}

/// Opaque handle to a renderer-owned texture: an `(owner, slot)` pair identifying one entry in the
/// renderer's texture store. `Copy`; valid for the lifetime of the renderer that produced it. Fields
/// are private to this module (mirrors `mesh::MeshId`) — nothing outside `texture.rs` can construct
/// a `TextureId` directly.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TextureId {
    owner: u64,
    slot: usize,
}

impl TextureId {
    /// Construct a `TextureId` for a renderer's `owner` id and a slot in its texture store. The sole
    /// constructor — called from `mod.rs`'s `create_texture` (the production path) and from this
    /// module's own tests.
    pub(super) fn new(owner: u64, slot: usize) -> Self {
        Self { owner, slot }
    }
}

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

/// Resolve `options.mipmaps` into a concrete `MipPolicy` for an R8G8B8A8_SRGB texture of
/// `width`x`height`, querying the selected device's format/image-format support as needed and
/// feeding the results to the pure `mip::select_mip_policy` decision helper (spec 014's
/// "Capability checks and explicit failure policy"). The image-format-properties query — the only
/// one that needs the real usage/extent/mip-count, and the only one that can return
/// `ERROR_FORMAT_NOT_SUPPORTED` — is skipped unless mips are actually requested for a texture
/// whose computed level count exceeds one; a 1x1 (or otherwise single-level) `Generate` request
/// needs no blit and so needs no capability check, matching `select_mip_policy`'s own bypass.
fn select_mip_policy_for(
    instance: &VulkanInstance,
    device: &VulkanDevice,
    width: u32,
    height: u32,
    options: TextureOptions,
) -> Result<mip::MipPolicy> {
    let requested = options.mipmaps;
    let format_features = unsafe {
        instance.instance.get_physical_device_format_properties(
            device.physical_device,
            vk::Format::R8G8B8A8_SRGB,
        )
    }
    .optimal_tiling_features;

    let image_format_properties = if requested == Mipmaps::Generate {
        let level_count = mip::mip_level_count(width, height)?;
        if level_count > 1 {
            let usage = vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC;
            match unsafe {
                instance
                    .instance
                    .get_physical_device_image_format_properties(
                        device.physical_device,
                        vk::Format::R8G8B8A8_SRGB,
                        vk::ImageType::TYPE_2D,
                        vk::ImageTiling::OPTIMAL,
                        usage,
                        vk::ImageCreateFlags::empty(),
                    )
            } {
                Ok(props) => Some(props),
                Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => None,
                Err(error) => return Err(VulkanError::from(error)),
            }
        } else {
            None
        }
    } else {
        None
    };

    mip::select_mip_policy(
        requested,
        width,
        height,
        format_features,
        image_format_properties,
    )
}

pub(crate) struct StoredTexture {
    image: SampledImage,
    pool: vk::DescriptorPool,
    pub(super) set: vk::DescriptorSet,
}

impl StoredTexture {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        instance: &VulkanInstance,
        layout: vk::DescriptorSetLayout,
        rgba8: &[u8],
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> Result<Self> {
        let policy = select_mip_policy_for(instance, device, width, height, options)?;
        // SampledImage::new_rgba8_srgb validates and rolls back its own partial resources on error.
        let image = SampledImage::new_rgba8_srgb(
            device,
            allocator,
            vk::Extent2D { width, height },
            rgba8,
            policy,
        )?;
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

/// The descriptor set bound at set 1 for an object: `white` when the object has no texture
/// (infallible — there is always a sensible default), or the stored texture's set when `texture` is
/// `Some` and resolves against `my_owner`/`texture_sets` (the identity check shared with
/// `mesh::resolve_mesh`, via `handle::resolve_handle`). Unlike `None`, a stale/foreign
/// `Some(TextureId)` — wrong owner or an out-of-range slot — is rejected with an error rather than
/// silently falling back to white.
pub(super) fn resolve_texture_set(
    texture: Option<TextureId>,
    my_owner: u64,
    texture_sets: &[vk::DescriptorSet],
    white: vk::DescriptorSet,
) -> Result<vk::DescriptorSet> {
    match texture {
        None => Ok(white),
        Some(id) => match handle::resolve_handle(id.owner, id.slot, my_owner, texture_sets.len()) {
            Some(slot) => Ok(texture_sets[slot]),
            None => Err(VulkanError::ValidationError(
                "texture id does not belong to this renderer or is out of range".into(),
            )),
        },
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
    fn texture_options_default_is_disabled() {
        assert_eq!(TextureOptions::default().mipmaps, Mipmaps::Disabled);
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
    fn resolve_texture_set_rejects_foreign_or_out_of_range_handles() {
        let white = set(0x10);
        let sets = [set(0xa0), set(0xb0)];
        let owner = handle::next_owner_id();

        // None always resolves to white, infallibly.
        assert_eq!(
            resolve_texture_set(None, owner, &sets, white).unwrap(),
            white
        );

        // A valid, correctly-owned Some(TextureId) resolves to the matching set.
        assert_eq!(
            resolve_texture_set(Some(TextureId::new(owner, 0)), owner, &sets, white).unwrap(),
            sets[0]
        );
        assert_eq!(
            resolve_texture_set(Some(TextureId::new(owner, 1)), owner, &sets, white).unwrap(),
            sets[1]
        );

        // An out-of-range slot with the correct owner is rejected.
        assert!(resolve_texture_set(Some(TextureId::new(owner, 2)), owner, &sets, white).is_err());

        // A mismatched owner with an otherwise in-range slot is rejected.
        let other = handle::next_owner_id();
        assert!(resolve_texture_set(Some(TextureId::new(owner, 0)), other, &sets, white).is_err());
    }
}
