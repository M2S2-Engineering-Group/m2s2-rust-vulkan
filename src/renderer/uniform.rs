// Per-frame uniform buffers + descriptor sets: the MVP matrices reaching the vertex shader. One
// uniform buffer and one descriptor set PER FRAME-IN-FLIGHT, so the CPU can rewrite frame N's copy
// while the GPU still reads frame N-1's. See specs/006-uniforms-depth-camera.md.

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::Buffer;
use crate::renderer::command::FRAMES_IN_FLIGHT;
use crate::renderer::device::VulkanDevice;
use ash::vk;

/// The shader-visible uniform block: model/view/proj kept separate (not a pre-multiplied MVP) so the
/// simulation side produces a world-space model matrix without knowing the camera. `#[repr(C)]` +
/// `Pod` so it uploads byte-for-byte; three `mat4`s are naturally std140-compatible (each 64 bytes,
/// 16-byte aligned, no padding).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

/// The binding index the UBO occupies in descriptor set 0 (must match the shader's
/// `layout(set = 0, binding = 0)`).
const UBO_BINDING: u32 = 0;

/// Per-frame-in-flight uniform buffers and the descriptor sets that bind them. Not `Drop`: the
/// buffers' allocations need `&mut Allocator` to free, so call `destroy` explicitly (see `Buffer`).
pub(crate) struct Uniforms {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    buffers: [Buffer; FRAMES_IN_FLIGHT], // host-visible, persistently mapped
}

impl Uniforms {
    pub(crate) fn new(device: &VulkanDevice, allocator: &mut Allocator) -> Result<Self> {
        let dev = &device.device;

        // set = 0: a single uniform buffer, read in the vertex stage.
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            dev.create_descriptor_set_layout(&layout_info, None)
                .map_err(VulkanError::from)?
        };

        // Pool sized for exactly one set (one uniform buffer) per frame-in-flight.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT as u32)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(FRAMES_IN_FLIGHT as u32);
        let descriptor_pool = unsafe {
            dev.create_descriptor_pool(&pool_info, None)
                .map_err(VulkanError::from)?
        };

        // One host-visible mapped buffer per frame-in-flight, each big enough for one UBO.
        let size = std::mem::size_of::<UniformBufferObject>() as u64;
        let mut buffers_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            buffers_vec.push(Buffer::new_host_visible_mapped(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?);
        }
        let buffers: [Buffer; FRAMES_IN_FLIGHT] = buffers_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("pushed exactly FRAMES_IN_FLIGHT buffers"));

        // Allocate one descriptor set per frame (same layout), then point each at its own buffer.
        let set_layouts = [descriptor_set_layout; FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = unsafe {
            dev.allocate_descriptor_sets(&alloc_info)
                .map_err(VulkanError::from)?
        }
        .try_into()
        .expect("allocate_descriptor_sets returns FRAMES_IN_FLIGHT sets");

        for i in 0..FRAMES_IN_FLIGHT {
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].buffer)
                .offset(0)
                .range(size)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(UBO_BINDING)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info);
            unsafe { dev.update_descriptor_sets(&[write], &[]) };
        }

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            buffers,
        })
    }

    /// Write `ubo` into frame `frame`'s mapped uniform buffer. The caller must have waited on that
    /// frame's in-flight fence first (the render loop does), so the GPU isn't mid-read.
    pub(crate) fn update(&self, frame: usize, ubo: &UniformBufferObject) {
        self.buffers[frame].write_mapped(bytemuck::bytes_of(ubo));
    }

    pub(crate) fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub(crate) fn descriptor_set(&self, frame: usize) -> vk::DescriptorSet {
        self.descriptor_sets[frame]
    }

    /// Free the mapped buffers (via `allocator`) and destroy the pool + layout (via `device`). The
    /// pool destroy implicitly frees the descriptor sets. Must run before the `Allocator` is dropped.
    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator) {
        for buffer in &mut self.buffers {
            buffer.destroy(device, allocator);
        }
        unsafe {
            device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_buffer_object_is_three_tightly_packed_mat4s() {
        // Three mat4 (64 bytes each), no padding — guards the std140 layout assumption and catches
        // an accidental field-type change.
        assert_eq!(std::mem::size_of::<UniformBufferObject>(), 3 * 64);
    }
}
