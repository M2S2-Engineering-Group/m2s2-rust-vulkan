//! Caller-facing directional light and the renderer-owned light uniform (descriptor set 2, fragment
//! stage). Per-pixel Lambert diffuse + ambient; structured so point/spot lights extend it later.

use crate::error::{Result, VulkanError};
use crate::math::Vec3;
use crate::renderer::allocator::Allocator;
use crate::renderer::device::VulkanDevice;
use crate::renderer::per_frame_uniform::PerFrameUniform;
use ash::vk;

/// A single directional ("sun") light in world space. Ergonomic value type; converted to the std140
/// GPU layout (`LightUniforms`) internally.
#[derive(Copy, Clone, Debug)]
pub struct DirectionalLight {
    /// The direction the light travels (sun rays), world space. Need not be normalized; the shader
    /// normalizes. Components must be finite and squared length must be finite and at least
    /// `f32::MIN_POSITIVE`; at least one squared component must also be normal to avoid GPU
    /// subnormal flushing. The surface-to-light vector used for shading is `-direction`.
    pub direction: Vec3,
    /// Finite, nonnegative diffuse light color/intensity (values above one are allowed). White (`1,1,1`) is a neutral full-strength sun.
    pub color: Vec3,
    /// Finite, nonnegative ambient color added to every fragment regardless of orientation, so the unlit side is dim
    /// rather than pure black.
    pub ambient: Vec3,
}

impl DirectionalLight {
    pub(crate) fn set_validated(&mut self, next: Self) -> Result<()> {
        let length_squared = (0..3)
            .map(|i| next.direction[i] * next.direction[i])
            .sum::<f32>();
        let has_normal_component =
            (0..3).any(|i| (next.direction[i] * next.direction[i]).is_normal());
        if !(0..3).all(|i| next.direction[i].is_finite())
            || !has_normal_component
            || !length_squared.is_finite()
            || length_squared < f32::MIN_POSITIVE
        {
            return Err(VulkanError::ValidationError(
                "light direction must be finite with a normalizable nonzero length".into(),
            ));
        }
        if !(0..3).all(|i| {
            next.color[i].is_finite()
                && next.color[i] >= 0.0
                && next.ambient[i].is_finite()
                && next.ambient[i] >= 0.0
        }) {
            return Err(VulkanError::ValidationError(
                "light color and ambient must be finite and nonnegative".into(),
            ));
        }
        *self = next;
        Ok(())
    }
}

impl Default for DirectionalLight {
    /// A neutral sun shining down-forward-right with a mild ambient floor.
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.5, -1.0, -0.3),
            color: Vec3::new(1.0, 1.0, 1.0),
            ambient: Vec3::new(0.15, 0.15, 0.15),
        }
    }
}

/// std140 mirror of the fragment shader's `LightUniforms` block: three `vec3`s, each 16-byte aligned,
/// so each carries an explicit trailing `f32` of padding. Size 48, offsets 0/16/32.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightUniforms {
    pub direction: Vec3,
    pub _pad0: f32,
    pub color: Vec3,
    pub _pad1: f32,
    pub ambient: Vec3,
    pub _pad2: f32,
}

impl From<&DirectionalLight> for LightUniforms {
    fn from(light: &DirectionalLight) -> Self {
        Self {
            direction: light.direction,
            _pad0: 0.0,
            color: light.color,
            _pad1: 0.0,
            ambient: light.ambient,
            _pad2: 0.0,
        }
    }
}

/// The binding index the light UBO occupies in descriptor set 2 (must match the fragment shader's
/// `layout(set = 2, binding = 0)`).
const LIGHT_BINDING: u32 = 0;

/// The light uniform: a `PerFrameUniform` carrying `LightUniforms` at `LIGHT_BINDING`, fragment stage.
pub(crate) struct Light {
    inner: PerFrameUniform,
}

impl Light {
    pub(crate) fn new(device: &VulkanDevice, allocator: &mut Allocator) -> Result<Self> {
        let inner = PerFrameUniform::new(
            device,
            allocator,
            std::mem::size_of::<LightUniforms>() as u64,
            LIGHT_BINDING,
            vk::ShaderStageFlags::FRAGMENT,
        )?;
        Ok(Self { inner })
    }

    /// Write `light` into frame `frame`'s mapped buffer (converted to std140). The caller must have
    /// waited on that frame's in-flight fence first, so the GPU isn't mid-read.
    pub(crate) fn update(&mut self, frame: usize, light: &DirectionalLight) -> Result<()> {
        self.inner
            .update(frame, bytemuck::bytes_of(&LightUniforms::from(light)))
    }

    pub(crate) fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.inner.descriptor_set_layout()
    }

    pub(crate) fn descriptor_set(&self, frame: usize) -> vk::DescriptorSet {
        self.inner.descriptor_set(frame)
    }

    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator) {
        self.inner.destroy(device, allocator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn setter_accepts_default_and_hdr_intensity() {
        let mut current = DirectionalLight::default();
        current.set_validated(DirectionalLight::default()).unwrap();
        let next = DirectionalLight {
            direction: Vec3::new(2.0, 0.0, 0.0),
            color: Vec3::new(4.0, 0.0, 1.0),
            ambient: Vec3::new(0.0, 2.0, 0.0),
        };
        current.set_validated(next).unwrap();
        assert_eq!(current.direction, next.direction);
        assert_eq!(current.color, next.color);
        assert_eq!(current.ambient, next.ambient);
    }

    #[test]
    fn setter_rejects_invalid_directions_without_changing_light() {
        for direction in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(f32::MIN_POSITIVE, 0.0, 0.0),
            Vec3::new(1e-19, 1e-19, 1e-19),
            Vec3::new(f32::MAX, 0.0, 0.0),
            Vec3::new(f32::NAN, 1.0, 0.0),
            Vec3::new(0.0, f32::INFINITY, 0.0),
            Vec3::new(0.0, 0.0, f32::NEG_INFINITY),
        ] {
            let mut current = DirectionalLight::default();
            let original = current;
            assert!(matches!(
                current.set_validated(DirectionalLight {
                    direction,
                    ..original
                }),
                Err(VulkanError::ValidationError(_))
            ));
            assert_eq!(current.direction, original.direction);
            assert_eq!(current.color, original.color);
            assert_eq!(current.ambient, original.ambient);
        }
    }

    #[test]
    fn setter_rejects_invalid_color_and_ambient_components() {
        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for component in 0..3 {
                for ambient in [false, true] {
                    let mut current = DirectionalLight::default();
                    let original = current;
                    let mut next = current;
                    if ambient {
                        next.ambient[component] = invalid;
                    } else {
                        next.color[component] = invalid;
                    }
                    assert!(current.set_validated(next).is_err());
                    assert_eq!(current.direction, original.direction);
                    assert_eq!(current.color, original.color);
                    assert_eq!(current.ambient, original.ambient);
                }
            }
        }
    }

    #[test]
    fn light_uniforms_match_std140_layout() {
        assert_eq!(std::mem::size_of::<LightUniforms>(), 48);
        assert_eq!(std::mem::offset_of!(LightUniforms, direction), 0);
        assert_eq!(std::mem::offset_of!(LightUniforms, color), 16);
        assert_eq!(std::mem::offset_of!(LightUniforms, ambient), 32);
    }

    #[test]
    fn directional_light_converts_fields_and_zeroes_padding() {
        let light = DirectionalLight {
            direction: Vec3::new(1.0, -2.0, 3.0),
            color: Vec3::new(0.4, 0.5, 0.6),
            ambient: Vec3::new(0.1, 0.1, 0.1),
        };
        let u = LightUniforms::from(&light);
        assert_eq!(u.direction, light.direction);
        assert_eq!(u.color, light.color);
        assert_eq!(u.ambient, light.ambient);
        assert_eq!((u._pad0, u._pad1, u._pad2), (0.0, 0.0, 0.0));
    }

    #[test]
    fn default_light_is_a_sun_with_ambient_floor() {
        let d = DirectionalLight::default();
        assert!(d.direction != Vec3::new(0.0, 0.0, 0.0));
        assert!(d.color != Vec3::new(0.0, 0.0, 0.0));
        assert!(d.ambient[0] > 0.0 && d.ambient[0] < 1.0);
    }
}
