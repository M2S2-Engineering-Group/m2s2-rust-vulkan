// Per-frame camera uniforms + descriptor sets: the camera matrices reaching the vertex shader. A thin
// typed wrapper over the shared `PerFrameUniform` helper (one uniform buffer + descriptor set per
// frame-in-flight). See specs/007-multiple-objects-and-push-constants.md.

use crate::error::{Result, VulkanError};
use crate::math::{Mat4, Vec3};
use crate::renderer::allocator::Allocator;
use crate::renderer::device::VulkanDevice;
use crate::renderer::per_frame_uniform::PerFrameUniform;
use ash::vk;

/// Caller-facing camera: view/projection matrices plus the world-space eye position (needed by the
/// fragment shader for specular lighting's view-direction term; spec 013). Converted to the std140
/// GPU layout (`CameraUniforms`) internally.
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
}

/// Validate a `Camera` before any GPU work. View/projection validity remains the caller's existing
/// contract (unchanged by spec 013); only the new `position` field is checked here.
pub(crate) fn validate_camera(camera: &Camera) -> Result<()> {
    if !(0..3).all(|i| camera.position[i].is_finite()) {
        return Err(VulkanError::ValidationError(
            "camera position must be finite".into(),
        ));
    }
    Ok(())
}

/// Per-frame view/projection matrices plus the world-space camera position, matching the shader's
/// std140 camera block; model transforms use push constants (spec 007). `position` is a `vec3` with
/// std140's implied 16-byte alignment, hence the trailing `_pad`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
    pub _pad: f32,
}

impl From<&Camera> for CameraUniforms {
    fn from(camera: &Camera) -> Self {
        Self {
            view: camera.view,
            proj: camera.proj,
            position: camera.position,
            _pad: 0.0,
        }
    }
}

/// The binding index the UBO occupies in descriptor set 0 (must match the shader's
/// `layout(set = 0, binding = 0)`).
const UBO_BINDING: u32 = 0;

/// The camera uniform: a `PerFrameUniform` carrying `CameraUniforms` at `UBO_BINDING`, vertex stage.
pub(crate) struct Uniforms {
    inner: PerFrameUniform,
}

impl Uniforms {
    pub(crate) fn new(device: &VulkanDevice, allocator: &mut Allocator) -> Result<Self> {
        let inner = PerFrameUniform::new(
            device,
            allocator,
            std::mem::size_of::<CameraUniforms>() as u64,
            UBO_BINDING,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        )?;
        Ok(Self { inner })
    }

    /// Write `ubo` into frame `frame`'s mapped uniform buffer. The caller must have waited on that
    /// frame's in-flight fence first (the render loop does), so the GPU isn't mid-read.
    pub(crate) fn update(&mut self, frame: usize, ubo: &CameraUniforms) -> Result<()> {
        self.inner.update(frame, bytemuck::bytes_of(ubo))
    }

    pub(crate) fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.inner.descriptor_set_layout()
    }

    pub(crate) fn descriptor_set(&self, frame: usize) -> vk::DescriptorSet {
        self.inner.descriptor_set(frame)
    }

    /// Free the mapped buffers (via `allocator`) and destroy the pool + layout (via `device`). Must
    /// run before the `Allocator` is dropped.
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        self.inner.destroy(device, allocator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_uniforms_match_shader_layout() {
        assert_eq!(std::mem::size_of::<CameraUniforms>(), 144);
        assert_eq!(std::mem::offset_of!(CameraUniforms, view), 0);
        assert_eq!(std::mem::offset_of!(CameraUniforms, proj), 64);
        assert_eq!(std::mem::offset_of!(CameraUniforms, position), 128);
        assert_eq!(std::mem::offset_of!(CameraUniforms, _pad), 140);
    }

    #[test]
    fn camera_converts_fields_and_zeroes_padding() {
        let camera = Camera {
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            position: Vec3::new(1.0, 2.0, 3.0),
        };
        let u = CameraUniforms::from(&camera);
        assert_eq!(
            (u.view, u.proj, u.position, u._pad),
            (camera.view, camera.proj, camera.position, 0.0)
        );
    }

    #[test]
    fn validate_camera_rejects_non_finite_position() {
        let base = Camera {
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            position: Vec3::new(0.0, 0.0, 0.0),
        };
        assert!(validate_camera(&base).is_ok());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for i in 0..3 {
                let mut c = base;
                c.position[i] = bad;
                assert!(validate_camera(&c).is_err());
            }
        }
    }
}
