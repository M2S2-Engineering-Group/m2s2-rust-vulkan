// Startup-only submission shared by buffer and sampled-image uploads.
use super::device::VulkanDevice;
use crate::error::{Result, VulkanError};
use ash::vk;

pub(super) fn submit_once(
    device: &VulkanDevice,
    record: impl FnOnce(&ash::Device, vk::CommandBuffer),
) -> Result<()> {
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
        record(dev, cmd);
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
