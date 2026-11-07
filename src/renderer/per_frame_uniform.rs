// A per-frame-in-flight, host-visible, persistently-mapped uniform buffer bound through a
// single-binding descriptor set. One buffer + one descriptor set PER FRAME-IN-FLIGHT, so the CPU can
// rewrite frame N's copy while the GPU still reads frame N-1's. Shared by the camera uniform
// (`uniform.rs`) and the light uniform (`light.rs`) — parameterized by payload size, binding index,
// and shader stage so neither duplicates this machinery.

use crate::error::{Result, VulkanError};
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::Buffer;
use crate::renderer::command::FRAMES_IN_FLIGHT;
use crate::renderer::device::VulkanDevice;
use ash::vk;

/// Per-frame-in-flight uniform buffers and the single-binding descriptor sets that bind them. Not
/// `Drop`: the buffers' allocations need `&mut Allocator` to free, so call `destroy` explicitly.
pub(crate) struct PerFrameUniform {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    buffers: [Buffer; FRAMES_IN_FLIGHT], // host-visible, persistently mapped
}

impl PerFrameUniform {
    /// Build a set with one `UNIFORM_BUFFER` binding (`binding`, visible to `stage`), backed by one
    /// mapped buffer of `size` bytes per frame-in-flight.
    pub(crate) fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        size: u64,
        binding: u32,
        stage: vk::ShaderStageFlags,
    ) -> Result<Self> {
        let dev = &device.device;

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(stage)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            dev.create_descriptor_set_layout(&layout_info, None)
                .map_err(VulkanError::from)?
        };

        let layout_cleanup = super::cleanup::Cleanup::new(|| unsafe {
            dev.destroy_descriptor_set_layout(descriptor_set_layout, None)
        });
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

        let pool_cleanup = super::cleanup::Cleanup::new(|| unsafe {
            dev.destroy_descriptor_pool(descriptor_pool, None)
        });
        // One host-visible mapped buffer per frame-in-flight, each big enough for one payload.
        let mut buffers_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let buffer = Buffer::new_host_visible_mapped(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            );
            match buffer {
                Ok(buffer) => buffers_vec.push(buffer),
                Err(error) => {
                    for buffer in &mut buffers_vec {
                        buffer.destroy(device, allocator);
                    }
                    return Err(error);
                }
            }
        }
        let mut buffers: [Buffer; FRAMES_IN_FLIGHT] = buffers_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("pushed exactly FRAMES_IN_FLIGHT buffers"));

        // Allocate one descriptor set per frame (same layout), then point each at its own buffer.
        let set_layouts = [descriptor_set_layout; FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let sets = match unsafe { dev.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets,
            Err(error) => {
                for buffer in &mut buffers {
                    buffer.destroy(device, allocator);
                }
                return Err(error.into());
            }
        };
        let descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = sets
            .try_into()
            .expect("allocate_descriptor_sets returns FRAMES_IN_FLIGHT sets");

        for i in 0..FRAMES_IN_FLIGHT {
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].buffer)
                .offset(0)
                .range(size)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(binding)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info);
            unsafe { dev.update_descriptor_sets(&[write], &[]) };
        }

        pool_cleanup.disarm();
        layout_cleanup.disarm();
        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            buffers,
        })
    }

    /// Write `bytes` into frame `frame`'s mapped buffer. The caller must have waited on that frame's
    /// in-flight fence first (the render loop does), so the GPU isn't mid-read.
    pub(crate) fn update(&self, frame: usize, bytes: &[u8]) {
        self.buffers[frame].write_mapped(bytes);
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
