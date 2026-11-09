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

/// A single point light in world space. Ergonomic value type; converted to the std140 GPU layout
/// (`PointLightUniforms`) internally as part of `SceneLightUniforms`.
#[derive(Copy, Clone, Debug)]
pub struct PointLight {
    /// World-space position of the light. Components must be finite.
    pub position: Vec3,
    /// Finite, nonnegative diffuse light color/intensity (values above one are allowed).
    pub color: Vec3,
    /// Finite, positive attenuation range: distance at which the light's contribution falls to zero.
    pub range: f32,
}

/// Maximum number of simultaneous point lights the fragment shader's `SceneLightUniforms` block
/// supports. Matches the `points` array length baked into that struct's std140 layout.
pub const MAX_POINT_LIGHTS: usize = 8;

/// Validates the whole slice up front (length first, then every element) and returns on the first
/// violation without mutating anything. Callers that hold previous state (e.g. Task 2's
/// `set_point_lights`) must call this before touching their own stored list, so a rejected call
/// leaves prior state untouched.
///
/// `#[allow(dead_code)]`: no non-test caller exists yet — Task 2's `set_point_lights` is the first
/// one. Exercised today only by this module's `#[cfg(test)]` tests.
#[allow(dead_code)]
pub(crate) fn validate_point_lights(lights: &[PointLight]) -> Result<()> {
    if lights.len() > MAX_POINT_LIGHTS {
        return Err(VulkanError::ValidationError(format!(
            "at most {MAX_POINT_LIGHTS} point lights are supported, got {}",
            lights.len()
        )));
    }
    for light in lights {
        if !(0..3).all(|i| light.position[i].is_finite()) {
            return Err(VulkanError::ValidationError(
                "point light position must be finite".into(),
            ));
        }
        if !(0..3).all(|i| light.color[i].is_finite() && light.color[i] >= 0.0) {
            return Err(VulkanError::ValidationError(
                "point light color must be finite and nonnegative".into(),
            ));
        }
        if !light.range.is_finite() || light.range <= 0.0 {
            return Err(VulkanError::ValidationError(
                "point light range must be finite and positive".into(),
            ));
        }
    }
    Ok(())
}

/// std140 mirror of one entry in the fragment shader's `points` array: `vec3 position`, `float
/// range`, `vec3 color`, with an explicit trailing `f32` of padding so the struct's size (32) is a
/// multiple of its largest member's alignment (16), matching GLSL's array-stride rule. Offsets
/// 0/12/16, size 32.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointLightUniforms {
    pub position: Vec3,
    pub range: f32,
    pub color: Vec3,
    pub _pad: f32,
}

/// std140 mirror of the fragment shader's `SceneLightUniforms` block: the directional light's three
/// `vec3`s (byte-identical in layout to the old `LightUniforms`, so the shader's existing block still
/// reads correctly), a point-light count, and a fixed-size array of `PointLightUniforms`. Size 320,
/// offsets: `direction` 0, `color` 16, `ambient` 32, `point_count` 48, `points` 64.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SceneLightUniforms {
    pub direction: Vec3,
    pub _pad0: f32,
    pub color: Vec3,
    pub _pad1: f32,
    pub ambient: Vec3,
    pub _pad2: f32,
    pub point_count: u32,
    pub _pad3: [u32; 3],
    pub points: [PointLightUniforms; MAX_POINT_LIGHTS],
}

impl SceneLightUniforms {
    /// Builds the full 320-byte struct from a directional light and up to `MAX_POINT_LIGHTS` point
    /// lights, zeroing every unused point slot. `points.len()` is trusted to already be
    /// `<= MAX_POINT_LIGHTS` (callers such as Task 2's `set_point_lights` enforce that upstream via
    /// `validate_point_lights`).
    pub(crate) fn new(directional: &DirectionalLight, points: &[PointLight]) -> Self {
        debug_assert!(points.len() <= MAX_POINT_LIGHTS);
        let mut point_uniforms = [bytemuck::Zeroable::zeroed(); MAX_POINT_LIGHTS];
        for (slot, point) in point_uniforms.iter_mut().zip(points.iter()) {
            *slot = PointLightUniforms {
                position: point.position,
                range: point.range,
                color: point.color,
                _pad: 0.0,
            };
        }
        Self {
            direction: directional.direction,
            _pad0: 0.0,
            color: directional.color,
            _pad1: 0.0,
            ambient: directional.ambient,
            _pad2: 0.0,
            point_count: points.len() as u32,
            _pad3: [0; 3],
            points: point_uniforms,
        }
    }
}

/// The binding index the light UBO occupies in descriptor set 2 (must match the fragment shader's
/// `layout(set = 2, binding = 0)`).
const LIGHT_BINDING: u32 = 0;

/// Fails if `SceneLightUniforms` (320 bytes) would not fit within `limit`
/// (`maxUniformBufferRange`), shaped like `pipeline.rs`'s `validate_push_limit`.
pub(crate) fn validate_light_uniform_limit(limit: u32) -> Result<()> {
    let required = std::mem::size_of::<SceneLightUniforms>() as u32;
    if required > limit {
        return Err(VulkanError::InitializationError(format!(
            "scene light uniform requires {required} bytes, device supports {limit}",
        )));
    }
    Ok(())
}

/// The light uniform: a `PerFrameUniform` carrying `SceneLightUniforms` at `LIGHT_BINDING`, fragment
/// stage.
pub(crate) struct Light {
    inner: PerFrameUniform,
}

impl Light {
    pub(crate) fn new(device: &VulkanDevice, allocator: &mut Allocator) -> Result<Self> {
        validate_light_uniform_limit(device.max_uniform_buffer_range)?;
        let inner = PerFrameUniform::new(
            device,
            allocator,
            std::mem::size_of::<SceneLightUniforms>() as u64,
            LIGHT_BINDING,
            vk::ShaderStageFlags::FRAGMENT,
        )?;
        Ok(Self { inner })
    }

    /// Write `directional` and `points` into frame `frame`'s mapped buffer (converted to std140).
    /// The caller must have waited on that frame's in-flight fence first, so the GPU isn't mid-read.
    pub(crate) fn update(
        &mut self,
        frame: usize,
        directional: &DirectionalLight,
        points: &[PointLight],
    ) -> Result<()> {
        let scene = SceneLightUniforms::new(directional, points);
        self.inner.update(frame, bytemuck::bytes_of(&scene))
    }

    pub(crate) fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.inner.descriptor_set_layout()
    }

    pub(crate) fn descriptor_set(&self, frame: usize) -> vk::DescriptorSet {
        self.inner.descriptor_set(frame)
    }

    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
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
    fn point_light_uniforms_match_std140_layout() {
        assert_eq!(std::mem::size_of::<PointLightUniforms>(), 32);
        assert_eq!(std::mem::offset_of!(PointLightUniforms, position), 0);
        assert_eq!(std::mem::offset_of!(PointLightUniforms, range), 12);
        assert_eq!(std::mem::offset_of!(PointLightUniforms, color), 16);
    }

    #[test]
    fn scene_light_uniforms_match_std140_layout() {
        assert_eq!(std::mem::size_of::<SceneLightUniforms>(), 320);
        assert_eq!(std::mem::offset_of!(SceneLightUniforms, direction), 0);
        assert_eq!(std::mem::offset_of!(SceneLightUniforms, color), 16);
        assert_eq!(std::mem::offset_of!(SceneLightUniforms, ambient), 32);
        assert_eq!(std::mem::offset_of!(SceneLightUniforms, point_count), 48);
        assert_eq!(std::mem::offset_of!(SceneLightUniforms, points), 64);
    }

    #[test]
    fn directional_light_converts_fields_and_zeroes_padding() {
        let light = DirectionalLight {
            direction: Vec3::new(1.0, -2.0, 3.0),
            color: Vec3::new(0.4, 0.5, 0.6),
            ambient: Vec3::new(0.1, 0.1, 0.1),
        };
        let u = SceneLightUniforms::new(&light, &[]);
        assert_eq!(u.direction, light.direction);
        assert_eq!(u.color, light.color);
        assert_eq!(u.ambient, light.ambient);
        assert_eq!((u._pad0, u._pad1, u._pad2), (0.0, 0.0, 0.0));
        assert_eq!(u.point_count, 0);
    }

    fn valid_point_light() -> PointLight {
        PointLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::new(0.5, 0.5, 0.5),
            range: 10.0,
        }
    }

    #[test]
    fn validate_point_lights_accepts_empty_and_max_count() {
        assert!(validate_point_lights(&[]).is_ok());
        let lights = vec![valid_point_light(); MAX_POINT_LIGHTS];
        assert!(validate_point_lights(&lights).is_ok());
    }

    #[test]
    fn validate_point_lights_rejects_too_many() {
        let lights = vec![valid_point_light(); MAX_POINT_LIGHTS + 1];
        assert!(matches!(
            validate_point_lights(&lights),
            Err(VulkanError::ValidationError(_))
        ));
    }

    #[test]
    fn validate_point_lights_rejects_invalid_individual_fields() {
        let invalid_values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

        for &bad in &invalid_values {
            for component in 0..3 {
                let mut light = valid_point_light();
                light.position[component] = bad;
                assert!(validate_point_lights(&[light]).is_err());
            }
        }

        for &bad in &invalid_values {
            for component in 0..3 {
                let mut light = valid_point_light();
                light.color[component] = bad;
                assert!(validate_point_lights(&[light]).is_err());
            }
        }

        for component in 0..3 {
            let mut light = valid_point_light();
            light.color[component] = -1.0;
            assert!(validate_point_lights(&[light]).is_err());
        }

        for &bad in &invalid_values {
            let mut light = valid_point_light();
            light.range = bad;
            assert!(validate_point_lights(&[light]).is_err());
        }

        let mut light = valid_point_light();
        light.range = 0.0;
        assert!(validate_point_lights(&[light]).is_err());

        let mut light = valid_point_light();
        light.range = -1.0;
        assert!(validate_point_lights(&[light]).is_err());
    }

    #[test]
    fn validate_point_lights_rejects_whole_list_on_trailing_invalid_light() {
        let mut lights = vec![valid_point_light(); MAX_POINT_LIGHTS];
        let mut invalid = valid_point_light();
        invalid.range = -1.0;
        lights.push(invalid);
        assert_eq!(lights.len(), MAX_POINT_LIGHTS + 1);
        assert!(validate_point_lights(&lights).is_err());
    }

    #[test]
    fn scene_light_uniforms_builds_from_directional_and_points() {
        let directional = DirectionalLight::default();
        let points = [
            PointLight {
                position: Vec3::new(1.0, 2.0, 3.0),
                color: Vec3::new(0.1, 0.2, 0.3),
                range: 5.0,
            },
            PointLight {
                position: Vec3::new(4.0, 5.0, 6.0),
                color: Vec3::new(0.4, 0.5, 0.6),
                range: 6.0,
            },
            PointLight {
                position: Vec3::new(7.0, 8.0, 9.0),
                color: Vec3::new(0.7, 0.8, 0.9),
                range: 7.0,
            },
        ];
        let scene = SceneLightUniforms::new(&directional, &points);
        assert_eq!(scene.point_count, 3);
        for (i, point) in points.iter().enumerate() {
            assert_eq!(scene.points[i].position, point.position);
            assert_eq!(scene.points[i].range, point.range);
            assert_eq!(scene.points[i].color, point.color);
            assert_eq!(scene.points[i]._pad, 0.0);
        }
        for point in &scene.points[3..MAX_POINT_LIGHTS] {
            assert_eq!(point.position, Vec3::new(0.0, 0.0, 0.0));
            assert_eq!(point.range, 0.0);
            assert_eq!(point.color, Vec3::new(0.0, 0.0, 0.0));
            assert_eq!(point._pad, 0.0);
        }
    }

    #[test]
    fn validate_light_uniform_limit_matches_scene_light_uniforms_size() {
        assert!(validate_light_uniform_limit(320).is_ok());
        assert!(validate_light_uniform_limit(319).is_err());
        assert!(validate_light_uniform_limit(16384).is_ok());
    }

    #[test]
    fn default_light_is_a_sun_with_ambient_floor() {
        let d = DirectionalLight::default();
        assert!(d.direction != Vec3::new(0.0, 0.0, 0.0));
        assert!(d.color != Vec3::new(0.0, 0.0, 0.0));
        assert!(d.ambient[0] > 0.0 && d.ambient[0] < 1.0);
    }
}
