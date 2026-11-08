// Per-frame camera uniforms + descriptor sets: the camera matrices reaching the vertex shader. A thin
// typed wrapper over the shared `PerFrameUniform` helper (one uniform buffer + descriptor set per
// frame-in-flight). See specs/007-multiple-objects-and-push-constants.md.

use crate::error::Result;
use crate::math::Mat4;
use crate::renderer::allocator::Allocator;
use crate::renderer::device::VulkanDevice;
use crate::renderer::per_frame_uniform::PerFrameUniform;
use ash::vk;

/// Per-frame view/projection matrices. Two tightly packed row-major mat4 values match
/// the shader's std140 camera block; model transforms use push constants (spec 007).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniforms {
    pub view: Mat4,
    pub proj: Mat4,
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
            vk::ShaderStageFlags::VERTEX,
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
        assert_eq!(std::mem::size_of::<CameraUniforms>(), 128);
        assert_eq!(std::mem::offset_of!(CameraUniforms, view), 0);
        assert_eq!(std::mem::offset_of!(CameraUniforms, proj), 64);
    }
}
