// Vertex data types and GPU buffers. See specs/003-vertex-buffers.md.

use crate::error::{Result, VulkanError};
use crate::math::Vec3;
use crate::renderer::allocator::Allocator;
use crate::renderer::device::VulkanDevice;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// Index type for index buffers — `u16` (bound as `vk::IndexType::UINT16`).
pub type Index = u16;

/// A single vertex: a position and a color, both in clip space / RGB. `#[repr(C)]` + `Pod` so a
/// `&[Vertex]` can be uploaded to a GPU buffer byte-for-byte (see `Buffer`). `Vec3` is
/// `m2s2-math`'s `Vector3f32`, `Pod` under its `bytemuck` feature (identical layout to `[f32; 3]`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub color: Vec3,
}

impl Vertex {
    /// One vertex buffer, consumed one `Vertex` per vertex.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// Shader inputs: `location = 0` position, `location = 1` color — offsets derived from the
    /// struct so they can't drift from the field order.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, color) as u32),
        ]
    }
}

/// A device-local GPU buffer whose contents were uploaded once via a staging buffer. Not `Drop`:
/// freeing the allocation needs `&mut Allocator`, so call `destroy` explicitly before the
/// `Allocator` is torn down (see specs/003).
pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    allocation: Option<Allocation>,
    len: u32,
}

impl Buffer {
    /// Create a `DEVICE_LOCAL` buffer of `usage` and upload `data` into it via a transient
    /// host-visible staging buffer + a one-time `vkCmdCopyBuffer`.
    pub(crate) fn new_device_local<T: bytemuck::Pod>(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let size = bytes.len() as u64;

        // Staging buffer in host-visible memory; gpu-allocator maps it, so we can memcpy straight in.
        let (staging, mut staging_alloc) = create_buffer(
            device,
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging",
        )?;
        staging_alloc
            .mapped_slice_mut()
            .expect("CpuToGpu allocation is host-visible and mapped")[..bytes.len()]
            .copy_from_slice(bytes);

        let result = (|| {
            let (buffer, allocation) = create_buffer(
                device,
                allocator,
                size,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "device-local",
            )?;
            if let Err(error) = copy_buffer(device, staging, buffer, size) {
                unsafe { device.device.destroy_buffer(buffer, None) };
                let _ = allocator.free(allocation);
                return Err(error);
            }
            Ok(Self {
                buffer,
                allocation: Some(allocation),
                len: data.len() as u32,
            })
        })();
        unsafe { device.device.destroy_buffer(staging, None) };
        let _ = allocator.free(staging_alloc);
        result
    }

    /// Create a `HOST_VISIBLE` buffer that gpu-allocator keeps persistently mapped, for data the CPU
    /// rewrites every frame (uniform buffers). Unlike `new_device_local` there's no staging copy —
    /// the caller writes straight into the mapping via `write_mapped`. `len` is left 0 (not a draw
    /// source). See specs/006.
    pub(crate) fn new_host_visible_mapped(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let (buffer, allocation) = create_buffer(
            device,
            allocator,
            size,
            usage,
            MemoryLocation::CpuToGpu,
            "uniform",
        )?;
        Ok(Self {
            buffer,
            allocation: Some(allocation),
            len: 0,
        })
    }

    /// Copy `bytes` into this buffer's persistent mapping. Only valid on a buffer from
    /// `new_host_visible_mapped`. Takes `&self`: it writes through the allocation's mapped pointer
    /// (no `&mut` needed), so a per-frame uniform update doesn't require mutable access. The caller
    /// must ensure the GPU isn't still reading this buffer (the per-frame-in-flight fence handles
    /// that in the render loop).
    pub(crate) fn write_mapped(&self, bytes: &[u8]) {
        let ptr = self
            .allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .expect("host-visible mapped buffer has a mapping")
            .as_ptr() as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    }

    /// Element count (vertices or indices), for `cmd_draw` / `cmd_draw_indexed`.
    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    /// Destroy the buffer and return its allocation to the allocator. Must run before the
    /// `Allocator` is dropped.
    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe { device.device.destroy_buffer(self.buffer, None) };
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Create a `VkBuffer`, allocate memory of `location` for it, and bind them.
fn create_buffer(
    device: &VulkanDevice,
    allocator: &mut Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &str,
) -> Result<(vk::Buffer, Allocation)> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe {
        device
            .device
            .create_buffer(&info, None)
            .map_err(VulkanError::from)?
    };
    let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
    let allocation_result = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location,
        linear: true, // buffers are always linear
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    });
    let allocation = match allocation_result {
        Ok(allocation) => allocation,
        Err(error) => {
            unsafe { device.device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    let binding = unsafe {
        device
            .device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
    };
    if let Err(error) = binding {
        unsafe { device.device.destroy_buffer(buffer, None) };
        let _ = allocator.free(allocation);
        return Err(error.into());
    }
    Ok((buffer, allocation))
}

/// Copy `size` bytes `src -> dst` via a one-time command buffer on the graphics queue, waiting on
/// a fence for completion (startup-only; fine to block here).
fn copy_buffer(device: &VulkanDevice, src: vk::Buffer, dst: vk::Buffer, size: u64) -> Result<()> {
    let dev = &device.device;
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(device.graphics_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe {
        dev.create_command_pool(&pool_info, None)
            .map_err(VulkanError::from)?
    };
    let _pool_cleanup =
        super::cleanup::Cleanup::new(|| unsafe { dev.destroy_command_pool(pool, None) });
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe {
        dev.allocate_command_buffers(&alloc_info)
            .map_err(VulkanError::from)?
    }[0];

    unsafe {
        dev.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(VulkanError::from)?;
        dev.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy::default().size(size)]);
        dev.end_command_buffer(cmd).map_err(VulkanError::from)?;

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let fence = dev
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(VulkanError::from)?;
        let _fence_cleanup = super::cleanup::Cleanup::new(|| dev.destroy_fence(fence, None));
        dev.queue_submit(device.graphics_queue, &[submit], fence)
            .map_err(VulkanError::from)?;
        dev.wait_for_fences(&[fence], true, u64::MAX)
            .map_err(VulkanError::from)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_is_tightly_packed() {
        // position (Vec3 = 12 bytes) + color (Vec3 = 12 bytes), no padding.
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
    }

    #[test]
    fn binding_description_describes_one_per_vertex_buffer() {
        let b = Vertex::binding_description();
        assert_eq!(b.binding, 0);
        assert_eq!(b.stride, std::mem::size_of::<Vertex>() as u32);
        assert_eq!(b.input_rate, vk::VertexInputRate::VERTEX);
    }

    #[test]
    fn attribute_descriptions_map_position_then_color() {
        let a = Vertex::attribute_descriptions();
        // location 0 = position, at the start of the struct
        assert_eq!(a[0].location, 0);
        assert_eq!(a[0].binding, 0);
        assert_eq!(a[0].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(a[0].offset, 0);
        // location 1 = color, 12 bytes in (after the position Vec3)
        assert_eq!(a[1].location, 1);
        assert_eq!(a[1].binding, 0);
        assert_eq!(a[1].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(a[1].offset, 12);
    }
}
