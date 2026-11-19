/// Pure helper functions for mipmap-level planning: computing mip-level counts,
/// mip extents, and the sequence of blits needed to generate all levels from a base image.
/// No GPU types — just integer math and planning data structures.
use crate::error::{Result, VulkanError};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
