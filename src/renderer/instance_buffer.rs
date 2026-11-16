// Per-frame-mapped instance buffer for GPU instancing (plan 016, Task 1). One host-visible vertex
// buffer per slot (typically one per frame-in-flight, see `command::FRAMES_IN_FLIGHT`), each holding
// instance-rate `InstanceData` for that frame's instanced draw(s). Unlike `PerFrameUniform` this is a
// plain vertex buffer with no descriptor set at all, so it mirrors
// `Buffer::new_host_visible_mapped`/`write_mapped` directly rather than reusing that machinery — see
// `Light`/`Uniforms` for the equivalent per-frame-array-of-`Buffer` shape this follows.
//
// `#[allow(dead_code)]` throughout this module: no non-test caller exists yet. Task 3 of the
// gpu-instancing plan is the first real caller. Exercised today only by this module's `#[cfg(test)]`
// tests.

use crate::error::{Result, VulkanError};
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::Buffer;
use crate::renderer::device::VulkanDevice;
use crate::renderer::pipeline::InstanceData;
use ash::vk;

/// Maximum number of instances one `InstanceBuffer` slot can hold in a single `write` (and the count
/// each slot's buffer is sized for at construction).
#[allow(dead_code)]
pub(crate) const MAX_INSTANCES_PER_FRAME: usize = 4096;

/// Rejects a `write` whose instance count would exceed `MAX_INSTANCES_PER_FRAME`, before touching any
/// buffer. Pure so it's unit-testable without a GPU (same shape as `pipeline.rs`'s
/// `validate_push_limit` / `buffer.rs`'s `check_mapped_write`).
#[allow(dead_code)]
fn validate_instance_count(len: usize) -> Result<()> {
    if len > MAX_INSTANCES_PER_FRAME {
        return Err(VulkanError::ValidationError(format!(
            "instance write of {len} instances exceeds MAX_INSTANCES_PER_FRAME ({MAX_INSTANCES_PER_FRAME})",
        )));
    }
    Ok(())
}

/// One host-visible, persistently-mapped vertex buffer per slot (e.g. one per frame-in-flight), each
/// sized for `MAX_INSTANCES_PER_FRAME` `InstanceData` instances. Not `Drop`: freeing the allocations
/// needs `&mut Allocator`, so call `destroy` explicitly before the `Allocator` is torn down (same
/// contract as `buffer::Buffer`/`PerFrameUniform`).
#[allow(dead_code)]
pub(crate) struct InstanceBuffer {
    buffers: Vec<Buffer>,
}

impl InstanceBuffer {
    /// Build `slots` independent buffers (e.g. `FRAMES_IN_FLIGHT` of them), each big enough for
    /// `MAX_INSTANCES_PER_FRAME` `InstanceData` instances. On a mid-construction failure, every buffer
    /// built so far is destroyed before returning the error (same rollback shape as
    /// `PerFrameUniform::new`).
    #[allow(dead_code)]
    pub(crate) fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        slots: usize,
    ) -> Result<Self> {
        let size = (MAX_INSTANCES_PER_FRAME * std::mem::size_of::<InstanceData>()) as u64;
        let mut buffers = Vec::with_capacity(slots);
        for _ in 0..slots {
            match Buffer::new_host_visible_mapped(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            ) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for buffer in buffers {
                        buffer.destroy(device, allocator);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self { buffers })
    }

    /// Write `data` into slot `frame`'s mapped buffer. Rejects a slice longer than
    /// `MAX_INSTANCES_PER_FRAME` before touching memory. The caller must have waited on that slot's
    /// in-flight fence first, so the GPU isn't mid-read (same contract as
    /// `Uniforms::update`/`Light::update`).
    #[allow(dead_code)]
    pub(crate) fn write(&mut self, frame: usize, data: &[InstanceData]) -> Result<()> {
        validate_instance_count(data.len())?;
        self.buffers[frame].write_mapped(bytemuck::cast_slice(data))
    }

    /// The raw `vk::Buffer` handle for slot `frame`, to bind as an instance-rate vertex buffer.
    #[allow(dead_code)]
    pub(crate) fn buffer(&self, frame: usize) -> vk::Buffer {
        self.buffers[frame].buffer
    }

    /// Destroy every slot's buffer and return its allocation to `allocator`. Must run before the
    /// `Allocator` is dropped.
    #[allow(dead_code)]
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        for buffer in self.buffers {
            buffer.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `InstanceBuffer::new`/`write` need a real `VulkanDevice` + `Allocator` (GPU-backed), so they
    // aren't independently unit-testable without a driver (see CLAUDE.md's Quality gates section on
    // not touching a real Vulkan driver from an ungated test). Per the plan's Task 1 brief, this is
    // the allowed reduced scope: exercise the pure length-validation logic `write` relies on,
    // covering both the in-bounds and over-`MAX_INSTANCES_PER_FRAME` cases without GPU readback.
    #[test]
    fn validate_instance_count_accepts_zero_through_max() {
        assert!(validate_instance_count(0).is_ok());
        assert!(validate_instance_count(1).is_ok());
        assert!(validate_instance_count(MAX_INSTANCES_PER_FRAME).is_ok());
    }

    #[test]
    fn validate_instance_count_rejects_over_max() {
        assert!(matches!(
            validate_instance_count(MAX_INSTANCES_PER_FRAME + 1),
            Err(VulkanError::ValidationError(_))
        ));
    }
}
