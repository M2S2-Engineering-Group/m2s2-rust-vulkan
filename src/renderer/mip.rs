/// Pure helper functions for mipmap-level planning: computing mip-level counts,
/// mip extents, and the sequence of blits needed to generate all levels from a base image.
/// No GPU types — just integer math and planning data structures.
use crate::error::{Result, VulkanError};
use ash::vk;

/// Mipmap generation policy: disabled (single-level image) or enabled (multi-level generation).
/// Task 3 will move the *public* version of this enum to `texture.rs` and re-export from `lib.rs`;
/// this is a temporary placeholder for decision logic.
/// `#[allow(dead_code)]`: consumed by Task 3. Exercised today only by this module's tests.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Mipmaps {
    Disabled,
    Generate,
}

/// The result of a capability check and mipmap policy decision: either single-level, or
/// multi-level generation with the computed level count.
/// `#[allow(dead_code)]`: consumed by Task 3. Exercised today only by this module's tests.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MipPolicy {
    OneLevel,
    Generate { level_count: u32 },
}

/// Computes the number of mip levels needed to tile an image down to 1x1.
/// For a base image of `width × height`, this returns `floor(log2(max(width, height))) + 1`,
/// using bit operations to avoid floating-point.
///
/// Returns `Err(ValidationError)` if either dimension is 0.
///
/// `#[allow(dead_code)]`: consumed by Tasks 2 and 3. Exercised today only by this module's tests.
#[allow(dead_code)]
pub(crate) fn mip_level_count(width: u32, height: u32) -> Result<u32> {
    if width == 0 || height == 0 {
        return Err(VulkanError::ValidationError(
            "mipmap dimensions must be non-zero".into(),
        ));
    }

    let max_dim = width.max(height);
    // floor(log2(max_dim)) + 1 = u32::BITS - 1 - max_dim.leading_zeros() + 1
    let level_count = u32::BITS - 1 - max_dim.leading_zeros() + 1;
    Ok(level_count)
}

/// Computes the extent (width, height) of mip level `level` given a base image of
/// `base_width × base_height`. Each level is halved in both dimensions (rounded down),
/// but never reaches 0 in either dimension — the smallest is always 1×1.
///
/// `#[allow(dead_code)]`: consumed by Tasks 2 and 3. Exercised today only by this module's tests.
#[allow(dead_code)]
pub(crate) fn mip_extent(base_width: u32, base_height: u32, level: u32) -> (u32, u32) {
    let w = (base_width >> level).max(1);
    let h = (base_height >> level).max(1);
    (w, h)
}

/// One blit operation from one mip level to the next. Encodes the source and destination
/// level numbers and their respective extents, so the caller can record the actual
/// `vkCmdBlitImage` command.
///
/// `#[allow(dead_code)]`: consumed by Tasks 2 and 3. Exercised today only by this module's tests.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlitStep {
    pub src_level: u32,
    pub dst_level: u32,
    pub src_extent: (u32, u32),
    pub dst_extent: (u32, u32),
}

/// Plans the sequence of blit steps needed to generate all mip levels from a base image.
/// Returns one `BlitStep` per level from level 1 to `level_count - 1`, with each step
/// blitting from the previous level to the next. Returns an empty `Vec` when `level_count <= 1`.
///
/// `#[allow(dead_code)]`: consumed by Tasks 2 and 3. Exercised today only by this module's tests.
#[allow(dead_code)]
pub(crate) fn plan_blits(base_width: u32, base_height: u32, level_count: u32) -> Vec<BlitStep> {
    if level_count <= 1 {
        return Vec::new();
    }

    let mut steps = Vec::with_capacity((level_count - 1) as usize);
    for i in 1..level_count {
        let src_level = i - 1;
        let dst_level = i;
        let src_extent = mip_extent(base_width, base_height, src_level);
        let dst_extent = mip_extent(base_width, base_height, dst_level);
        steps.push(BlitStep {
            src_level,
            dst_level,
            src_extent,
            dst_extent,
        });
    }
    steps
}

/// Decides whether to generate mipmaps based on the requested policy and device capabilities.
/// Returns a `MipPolicy` with the appropriate level count, or a `ValidationError` if
/// required capabilities are unsupported.
///
/// Logic:
/// - `Mipmaps::Disabled` → always `Ok(MipPolicy::OneLevel)`, no capability check.
/// - `Mipmaps::Generate` with computed `level_count == 1` → `Ok(MipPolicy::Generate { level_count: 1 })`,
///   no capability check (nothing to blit).
/// - `Mipmaps::Generate` with `level_count > 1` → require all of:
///   - `format_features` contains `BLIT_SRC | BLIT_DST | SAMPLED_IMAGE_FILTER_LINEAR | SAMPLED_IMAGE`
///   - `image_format_properties` is `Some(_)` with `max_mip_levels >= level_count` and
///     `max_extent` covering `(width, height)`
///   - If all present: `Ok(MipPolicy::Generate { level_count })`
///   - If any missing: `Err(ValidationError("..."))` naming which capability is missing
///
/// `#[allow(dead_code)]`: consumed by Task 3. Exercised today only by this module's tests.
#[allow(dead_code)]
pub(crate) fn select_mip_policy(
    requested: Mipmaps,
    width: u32,
    height: u32,
    format_features: vk::FormatFeatureFlags,
    image_format_properties: Option<vk::ImageFormatProperties>,
) -> Result<MipPolicy> {
    match requested {
        Mipmaps::Disabled => Ok(MipPolicy::OneLevel),
        Mipmaps::Generate => {
            let level_count = mip_level_count(width, height)?;
            if level_count == 1 {
                return Ok(MipPolicy::Generate { level_count: 1 });
            }

            // Check required format features
            if !format_features.contains(vk::FormatFeatureFlags::BLIT_SRC) {
                return Err(VulkanError::ValidationError(
                    "format does not support BLIT_SRC".into(),
                ));
            }

            if !format_features.contains(vk::FormatFeatureFlags::BLIT_DST) {
                return Err(VulkanError::ValidationError(
                    "format does not support BLIT_DST".into(),
                ));
            }

            if !format_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) {
                return Err(VulkanError::ValidationError(
                    "format does not support SAMPLED_IMAGE_FILTER_LINEAR".into(),
                ));
            }

            if !format_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
                return Err(VulkanError::ValidationError(
                    "format does not support SAMPLED_IMAGE".into(),
                ));
            }

            // Check image format properties
            let props = image_format_properties.ok_or_else(|| {
                VulkanError::ValidationError(
                    "get_physical_device_image_format_properties query returned VK_ERROR_FORMAT_NOT_SUPPORTED"
                        .into(),
                )
            })?;

            if props.max_mip_levels < level_count {
                return Err(VulkanError::ValidationError(format!(
                    "device supports only {} mip levels but {} are required",
                    props.max_mip_levels, level_count
                )));
            }

            if props.max_extent.width < width || props.max_extent.height < height {
                return Err(VulkanError::ValidationError(format!(
                    "device max_extent {:?} does not cover required ({}, {})",
                    props.max_extent, width, height
                )));
            }

            Ok(MipPolicy::Generate { level_count })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Task 1 tests
    #[test]
    fn mip_level_count_matches_spec_worked_examples() {
        assert_eq!(mip_level_count(1, 1).unwrap(), 1);
        assert_eq!(mip_level_count(8, 4).unwrap(), 4); // 8x4,4x2,2x1,1x1
        assert_eq!(mip_level_count(7, 3).unwrap(), 3); // 7x3,3x1,1x1
        assert_eq!(mip_level_count(1, 8).unwrap(), 4); // 1x8,1x4,1x2,1x1
    }

    #[test]
    fn mip_level_count_rejects_zero_dimensions() {
        assert!(mip_level_count(0, 4).is_err());
        assert!(mip_level_count(4, 0).is_err());
    }

    #[test]
    fn mip_extent_never_reaches_zero_in_either_dimension() {
        assert_eq!(mip_extent(8, 4, 0), (8, 4));
        assert_eq!(mip_extent(8, 4, 1), (4, 2));
        assert_eq!(mip_extent(8, 4, 2), (2, 1));
        assert_eq!(mip_extent(8, 4, 3), (1, 1));
        assert_eq!(mip_extent(7, 3, 2), (1, 1));
    }

    #[test]
    fn plan_blits_is_empty_for_one_level() {
        assert!(plan_blits(1, 1, 1).is_empty());
        assert!(plan_blits(8, 4, 1).is_empty());
    }

    #[test]
    fn plan_blits_emits_one_step_per_level_in_order_with_correct_extents() {
        let steps = plan_blits(8, 4, 4);
        assert_eq!(steps.len(), 3);
        assert_eq!((steps[0].src_level, steps[0].dst_level), (0, 1));
        assert_eq!((steps[0].src_extent, steps[0].dst_extent), ((8, 4), (4, 2)));
        assert_eq!((steps[1].src_level, steps[1].dst_level), (1, 2));
        assert_eq!((steps[1].src_extent, steps[1].dst_extent), ((4, 2), (2, 1)));
        assert_eq!((steps[2].src_level, steps[2].dst_level), (2, 3));
        assert_eq!((steps[2].src_extent, steps[2].dst_extent), ((2, 1), (1, 1)));
    }

    // Task 2 tests
    const FULL_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
        vk::FormatFeatureFlags::BLIT_SRC.as_raw()
            | vk::FormatFeatureFlags::BLIT_DST.as_raw()
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR.as_raw()
            | vk::FormatFeatureFlags::SAMPLED_IMAGE.as_raw(),
    );

    fn full_image_format_properties(level_count: u32) -> vk::ImageFormatProperties {
        vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 4096,
                height: 4096,
                depth: 1,
            },
            max_mip_levels: level_count,
            max_array_layers: 1,
            sample_counts: vk::SampleCountFlags::TYPE_1,
            max_resource_size: u64::MAX,
        }
    }

    #[test]
    fn disabled_always_succeeds_as_one_level_without_a_capability_check() {
        let policy = select_mip_policy(
            Mipmaps::Disabled,
            256,
            256,
            vk::FormatFeatureFlags::empty(),
            None,
        )
        .unwrap();
        assert!(matches!(policy, MipPolicy::OneLevel));
    }

    #[test]
    fn generate_on_1x1_succeeds_without_a_capability_check() {
        let policy = select_mip_policy(
            Mipmaps::Generate,
            1,
            1,
            vk::FormatFeatureFlags::empty(),
            None,
        )
        .unwrap();
        assert!(matches!(policy, MipPolicy::Generate { level_count: 1 }));
    }

    #[test]
    fn generate_with_full_support_succeeds_with_expected_level_count() {
        let policy = select_mip_policy(
            Mipmaps::Generate,
            256,
            256,
            FULL_FEATURES,
            Some(full_image_format_properties(9)),
        )
        .unwrap();
        assert!(matches!(policy, MipPolicy::Generate { level_count: 9 }));
    }

    #[test]
    fn generate_rejects_each_missing_required_format_feature_individually() {
        for missing in [
            vk::FormatFeatureFlags::BLIT_SRC,
            vk::FormatFeatureFlags::BLIT_DST,
            vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            vk::FormatFeatureFlags::SAMPLED_IMAGE,
        ] {
            let features =
                vk::FormatFeatureFlags::from_raw(FULL_FEATURES.as_raw() & !missing.as_raw());
            assert!(select_mip_policy(
                Mipmaps::Generate,
                256,
                256,
                features,
                Some(full_image_format_properties(9))
            )
            .is_err());
        }
    }

    #[test]
    fn generate_rejects_unsupported_image_format_properties_query() {
        assert!(select_mip_policy(Mipmaps::Generate, 256, 256, FULL_FEATURES, None).is_err());
    }

    #[test]
    fn generate_rejects_device_mip_level_cap_below_required_count() {
        assert!(select_mip_policy(
            Mipmaps::Generate,
            256,
            256,
            FULL_FEATURES,
            Some(full_image_format_properties(4))
        )
        .is_err()); // needs 9, device caps at 4
    }
}
