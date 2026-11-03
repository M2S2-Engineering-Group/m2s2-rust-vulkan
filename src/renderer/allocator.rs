// Thin wrapper over the `gpu-allocator` crate so the rest of the engine depends on this type, not
// the crate directly (Dependency Inversion — a future swap to a hand-written sub-allocator stays a
// one-file change). See specs/003-vertex-buffers.md ("Decision: gpu-allocator, wrapped").

use crate::error::{Result, VulkanError};
use crate::renderer::{device::VulkanDevice, instance::VulkanInstance};
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, Allocator as GpuAllocator, AllocatorCreateDesc,
};

/// Owns the backing `VkDeviceMemory` blocks (via `gpu-allocator`) and sub-allocates buffers out of
/// them. Must be dropped before the `VulkanDevice` it was built from — enforced by
/// `VulkanRenderer`'s explicit `ManuallyDrop` teardown order, not the type system.
pub struct Allocator {
    inner: GpuAllocator,
}

impl Allocator {
    pub fn new(instance: &VulkanInstance, device: &VulkanDevice) -> Result<Self> {
        let inner = GpuAllocator::new(&AllocatorCreateDesc {
            instance: instance.instance.clone(),
            device: device.device.clone(),
            physical_device: device.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        Ok(Self { inner })
    }

    pub(crate) fn allocate(&mut self, desc: &AllocationCreateDesc) -> Result<Allocation> {
        self.inner.allocate(desc).map_err(VulkanError::from)
    }

    pub(crate) fn free(&mut self, allocation: Allocation) -> Result<()> {
        self.inner.free(allocation).map_err(VulkanError::from)
    }
}
