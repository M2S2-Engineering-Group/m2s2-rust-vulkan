//! Bundles what a shader needs to shade one object (a texture, a base-color tint, and specular
//! parameters) into one type, independent of where that data came from (hand-written, glTF, or
//! any future asset source) — see spec 021.

use super::texture::{validate_specular, SpecularMaterial, TextureId};
use crate::error::Result;
use crate::math::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material {
    pub texture: Option<TextureId>,
    pub base_color: Vec3,
    pub specular: SpecularMaterial,
}

impl Default for Material {
    /// No texture, no tint (`base_color` white — a no-op multiply), and the pre-material-system
    /// `SpecularMaterial` default — byte-identical `InstanceData` to what `Object`'s old separate
    /// `texture`/`specular` fields produced before this spec.
    fn default() -> Self {
        Self {
            texture: None,
            base_color: Vec3::new(1.0, 1.0, 1.0),
            specular: SpecularMaterial::default(),
        }
    }
}

pub(crate) fn validate_material(material: &Material) -> Result<()> {
    validate_specular(&material.specular)?;
    if !(0..3).all(|i| material.base_color[i].is_finite() && material.base_color[i] >= 0.0) {
        return Err(crate::error::VulkanError::ValidationError(
            "material base_color must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_pre_material_object_defaults() {
        let m = Material::default();
        assert_eq!(m.texture, None);
        assert_eq!(m.base_color, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(m.specular, SpecularMaterial::default());
    }

    #[test]
    fn validate_material_rejects_non_finite_or_negative_base_color() {
        let base = Material::default();
        assert!(validate_material(&base).is_ok());
        for bad in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for i in 0..3 {
                let mut m = base;
                m.base_color[i] = bad;
                assert!(validate_material(&m).is_err());
            }
        }
    }

    #[test]
    fn validate_material_delegates_specular_validation() {
        let mut m = Material::default();
        m.specular.shininess = 0.0; // invalid per existing validate_specular rule
        assert!(validate_material(&m).is_err());
    }
}
