// Vertex data types and GPU buffers. See specs/003-vertex-buffers.md.

use crate::error::{Result, VulkanError};
use crate::math::{Vec2, Vec3};
use crate::renderer::allocator::Allocator;
use crate::renderer::device::VulkanDevice;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// Index type for index buffers — `u16` (bound as `vk::IndexType::UINT16`).
pub type Index = u16;

/// A single vertex: a position (clip/world space), an RGB color, a UV texture coordinate, and a
/// surface normal (used for lighting; must be non-zero — a `(0,0,0)` normal normalizes to NaN in the
/// shader and produces garbage shading). `#[repr(C)]` + `Pod` so a `&[Vertex]` uploads byte-for-byte.
/// `Vec2`/`Vec3` are `m2s2-math` vectors, `Pod` under its `bytemuck` feature (identical layout to
/// `[f32; N]`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub color: Vec3,
    pub uv: Vec2,
    pub normal: Vec3,
}

impl Vertex {
    /// One vertex buffer, consumed one `Vertex` per vertex.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// Shader inputs: `location = 0` position, `location = 1` color, `location = 2` uv,
    /// `location = 3` normal — offsets derived from the struct so they can't drift from the field
    /// order.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
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
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, uv) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(3)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, normal) as u32),
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
    /// Logical byte capacity the caller may write via `write_mapped` — the requested size, distinct
    /// from element count (`len`) and from any allocator padding behind the mapping.
    logical_capacity: usize,
}

/// Validate a `write_mapped` before touching memory: the buffer must be mapped and the write must fit
/// the logical capacity (allocator padding does not count). Pure so it is unit-testable without a GPU.
fn check_mapped_write(logical_capacity: usize, mapped: bool, len: usize) -> Result<()> {
    if !mapped {
        return Err(VulkanError::ValidationError(
            "write_mapped on an unmapped buffer".into(),
        ));
    }
    if len > logical_capacity {
        return Err(VulkanError::ValidationError(
            "write_mapped exceeds the buffer's logical capacity".into(),
        ));
    }
    Ok(())
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
                logical_capacity: bytes.len(),
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
            logical_capacity: size as usize,
        })
    }

    /// Copy `bytes` into this buffer's persistent mapping. Only valid on a buffer from
    /// `new_host_visible_mapped`. Takes `&mut self` so overlapping host writes through this API can't
    /// race, and rejects (before touching memory) an unmapped buffer or a write beyond the buffer's
    /// logical capacity. The caller must still ensure the GPU isn't reading this buffer — the
    /// per-frame-in-flight fence handles that in the render loop.
    pub(crate) fn write_mapped(&mut self, bytes: &[u8]) -> Result<()> {
        let mapped = self
            .allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .is_some();
        check_mapped_write(self.logical_capacity, mapped, bytes.len())?;
        let slice = self
            .allocation
            .as_mut()
            .and_then(|a| a.mapped_slice_mut())
            .expect("mapping checked above");
        slice[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Element count (vertices or indices), for `cmd_draw` / `cmd_draw_indexed`.
    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    /// Destroy the buffer and return its allocation to the allocator. Must run before the
    /// `Allocator` is dropped.
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe { device.device.destroy_buffer(self.buffer, None) };
        if let Some(allocation) = self.allocation {
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
    super::upload::submit_once(device, |dev, cmd| unsafe {
        dev.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy::default().size(size)]);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_write_bounds_and_mapping() {
        assert!(check_mapped_write(16, true, 0).is_ok()); // empty
        assert!(check_mapped_write(16, true, 16).is_ok()); // exact
        assert!(check_mapped_write(16, true, 8).is_ok()); // smaller
        assert!(check_mapped_write(16, true, 17).is_err()); // oversized
        assert!(check_mapped_write(16, false, 4).is_err()); // unmapped
    }

    #[test]
    fn vertex_is_tightly_packed() {
        assert_eq!(std::mem::size_of::<Vertex>(), 44);
        assert_eq!(std::mem::offset_of!(Vertex, position), 0);
        assert_eq!(std::mem::offset_of!(Vertex, color), 12);
        assert_eq!(std::mem::offset_of!(Vertex, uv), 24);
        assert_eq!(std::mem::offset_of!(Vertex, normal), 32);
    }

    #[test]
    fn binding_description_describes_one_per_vertex_buffer() {
        let b = Vertex::binding_description();
        assert_eq!(b.binding, 0);
        assert_eq!(b.stride, std::mem::size_of::<Vertex>() as u32);
        assert_eq!(b.input_rate, vk::VertexInputRate::VERTEX);
    }

    #[test]
    fn attribute_descriptions_map_position_color_then_uv_then_normal() {
        let attrs = Vertex::attribute_descriptions();
        assert_eq!(attrs.len(), 4);
        assert_eq!(attrs[0].location, 0);
        assert_eq!(attrs[0].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[1].location, 1);
        assert_eq!(attrs[1].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[1].offset, 12);
        assert_eq!(attrs[2].location, 2);
        assert_eq!(attrs[2].format, vk::Format::R32G32_SFLOAT);
        assert_eq!(attrs[2].offset, 24);
        assert_eq!(attrs[3].location, 3);
        assert_eq!(attrs[3].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[3].offset, 32);
    }
}
