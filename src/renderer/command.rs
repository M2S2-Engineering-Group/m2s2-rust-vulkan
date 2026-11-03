// Command pool/buffer management, synchronization, and the render loop. See
// specs/002-command-buffers-and-render-loop.md for the full design -- especially the "dynamic
// rendering gap" section, since this is the one part of the render loop that can't be a
// straight port of vulkan-tutorial.com's render-pass-based version.

use crate::error::{Result, VulkanError};
use crate::renderer::buffer::Buffer;
use crate::renderer::device::VulkanDevice;
use crate::renderer::instance::VulkanInstance;
use crate::renderer::pipeline::GraphicsPipeline;
use crate::renderer::swapchain::VulkanSwapchain;
use ash::vk;

pub const FRAMES_IN_FLIGHT: usize = 2;

/// Result of one `draw_frame`: whether it rendered normally or the swapchain went out of date (the
/// window resized) and the caller should recreate it.
pub enum FrameOutcome {
    Rendered,
    SwapchainOutOfDate,
}

pub struct FrameContext {
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    image_available_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    // One render-finished semaphore PER SWAPCHAIN IMAGE (not per frame-in-flight): present waits on
    // it with no fence to report when it's consumed, so a per-frame semaphore could be reused while
    // a prior present still references it (validation flags this). Indexed by acquired image index;
    // recreated when the image count changes on swapchain recreate.
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    current_frame: usize,
    // vkCmdBeginRendering/EndRendering are reached through the KHR extension loader, not
    // ash::Device: spec 001 enabled VK_KHR_dynamic_rendering on a 1.0 instance rather than
    // requiring Vulkan 1.3 core, so the core function pointers may be null on an extension-only
    // driver. The loader is cheap (a function-pointer table); holding it avoids rebuilding it
    // every frame. (Signature tweak vs. the spec's `new(&VulkanDevice)` — see Outcome.)
    dynamic_rendering: ash::khr::dynamic_rendering::Device,
    device: ash::Device, // cloned handle, kept only so Drop can destroy the fields above
}

impl FrameContext {
    pub fn new(
        instance: &VulkanInstance,
        device: &VulkanDevice,
        image_count: usize,
    ) -> Result<Self> {
        let dev = &device.device;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.graphics_queue_family)
            // RESET_COMMAND_BUFFER lets draw_frame reset one buffer per frame instead of the pool.
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            dev.create_command_pool(&pool_info, None)
                .map_err(VulkanError::from)?
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        let command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT] = unsafe {
            dev.allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::from)?
        }
        .try_into()
        .expect("allocate_command_buffers returns FRAMES_IN_FLIGHT buffers");

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // Created SIGNALED so the very first wait_for_fences (before any submit) returns
        // immediately instead of deadlocking.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = [vk::Semaphore::null(); FRAMES_IN_FLIGHT];
        let mut in_flight_fences = [vk::Fence::null(); FRAMES_IN_FLIGHT];
        for i in 0..FRAMES_IN_FLIGHT {
            unsafe {
                image_available_semaphores[i] = dev
                    .create_semaphore(&semaphore_info, None)
                    .map_err(VulkanError::from)?;
                in_flight_fences[i] = dev
                    .create_fence(&fence_info, None)
                    .map_err(VulkanError::from)?;
            }
        }
        // render-finished is per swapchain image (see the field comment).
        let render_finished_semaphores = Self::create_present_semaphores(dev, image_count)?;

        let dynamic_rendering = ash::khr::dynamic_rendering::Device::new(&instance.instance, dev);

        Ok(Self {
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            dynamic_rendering,
            device: dev.clone(),
        })
    }

    fn create_present_semaphores(dev: &ash::Device, count: usize) -> Result<Vec<vk::Semaphore>> {
        let info = vk::SemaphoreCreateInfo::default();
        (0..count)
            .map(|_| unsafe { dev.create_semaphore(&info, None).map_err(VulkanError::from) })
            .collect()
    }

    /// Recreate the per-swapchain-image render-finished semaphores — the image count can change on a
    /// swapchain recreate. The caller must have waited for the device to be idle.
    pub(crate) fn recreate_present_semaphores(
        &mut self,
        device: &VulkanDevice,
        image_count: usize,
    ) -> Result<()> {
        unsafe {
            for &semaphore in &self.render_finished_semaphores {
                device.device.destroy_semaphore(semaphore, None);
            }
        }
        self.render_finished_semaphores =
            Self::create_present_semaphores(&device.device, image_count)?;
        Ok(())
    }

    /// Runs one full frame against `swapchain`, using `pipeline` to draw. See
    /// specs/002's "Per-frame protocol" and "Recording sequence" sections.
    pub fn draw_frame(
        &mut self,
        device: &VulkanDevice,
        pipeline: &GraphicsPipeline,
        swapchain: &VulkanSwapchain,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
    ) -> Result<FrameOutcome> {
        let dev = &device.device;
        let frame = self.current_frame;
        let cmd = self.command_buffers[frame];
        let image_available = self.image_available_semaphores[frame];
        let fence = self.in_flight_fences[frame];
        let mut needs_recreate = false;

        // 1. Wait until this frame slot's previous submission has finished on the GPU.
        unsafe {
            dev.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(VulkanError::from)?;
        }

        // 2. Acquire the next swapchain image. image_available is signaled when it's safe to render
        //    into. Out-of-date = the swapchain no longer matches the window: skip this frame and
        //    report it so the renderer recreates. Return BEFORE resetting the fence (it must stay
        //    signaled, or step 1 next time around deadlocks). SUBOPTIMAL still renders but flags a
        //    recreate.
        let image_index = match unsafe {
            swapchain.swapchain_loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        } {
            Ok((index, suboptimal)) => {
                needs_recreate |= suboptimal;
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(FrameOutcome::SwapchainOutOfDate),
            Err(e) => return Err(VulkanError::from(e)),
        };
        // render-finished is per image (see field comment), indexed by the acquired image.
        let render_finished = self.render_finished_semaphores[image_index as usize];

        // 3. Only now that we're committed to submitting, unsignal the fence.
        unsafe {
            dev.reset_fences(&[fence]).map_err(VulkanError::from)?;
        }

        // 4. Record the command buffer.
        self.record(
            dev,
            cmd,
            pipeline,
            swapchain,
            image_index,
            vertex_buffer,
            index_buffer,
        )?;

        // 5. Submit: wait on image_available at the color-write stage, signal render_finished,
        //    and signal `fence` on completion so step 1 can reclaim this slot next cycle.
        let wait_semaphores = [image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [cmd];
        let signal_semaphores = [render_finished];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            dev.queue_submit(device.graphics_queue, &[submit_info], fence)
                .map_err(VulkanError::from)?;
        }

        // 6. Present: wait on render_finished so the display engine doesn't show a half-drawn
        //    image. Out-of-date/suboptimal flags a recreate (the renderer handles it next frame).
        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        match unsafe {
            swapchain
                .swapchain_loader
                .queue_present(device.present_queue, &present_info)
        } {
            Ok(suboptimal) => needs_recreate |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                needs_recreate = true;
            }
            Err(e) => return Err(VulkanError::from(e)),
        }

        // 7. Advance to the next frame-in-flight slot.
        self.current_frame = next_frame(frame);

        Ok(if needs_recreate {
            FrameOutcome::SwapchainOutOfDate
        } else {
            FrameOutcome::Rendered
        })
    }

    /// Records the draw into `cmd` for the acquired `image_index`. With dynamic rendering there are
    /// no subpass dependencies, so the swapchain image's layout transitions are explicit barriers.
    // Command recording genuinely needs all these handles; a mesh-bundle abstraction is a deferred
    // non-goal (specs/003), so allow the arg count rather than introduce one just for clippy.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        dev: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline: &GraphicsPipeline,
        swapchain: &VulkanSwapchain,
        image_index: u32,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
    ) -> Result<()> {
        let image = swapchain.images[image_index as usize];
        let image_view = swapchain.image_views[image_index as usize];
        let extent = swapchain.extent;

        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::from)?;
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(VulkanError::from)?;

            // a. UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL before we render into the image.
            let to_color = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_color],
            );

            // b. Begin dynamic rendering into the acquired image view, clearing to dark blue.
            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.08, 1.0],
                },
            };
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear);
            let color_attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments);
            self.dynamic_rendering
                .cmd_begin_rendering(cmd, &rendering_info);

            // c. Bind the pipeline.
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

            // d. Viewport + scissor are dynamic (spec 001), so they must be set every recording.
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            dev.cmd_set_viewport(cmd, 0, &[viewport]);
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            dev.cmd_set_scissor(cmd, 0, &[scissor]);

            // e. Bind the vertex + index buffers and draw the indexed geometry.
            dev.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer.buffer], &[0]);
            dev.cmd_bind_index_buffer(cmd, index_buffer.buffer, 0, vk::IndexType::UINT16);
            dev.cmd_draw_indexed(cmd, index_buffer.len(), 1, 0, 0, 0);

            // f. End rendering.
            self.dynamic_rendering.cmd_end_rendering(cmd);

            // g. COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR so the display engine can show it.
            let to_present = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );

            dev.end_command_buffer(cmd).map_err(VulkanError::from)?;
        }

        Ok(())
    }
}

fn next_frame(current: usize) -> usize {
    (current + 1) % FRAMES_IN_FLIGHT
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

impl Drop for FrameContext {
    fn drop(&mut self) {
        // Fences, then semaphores, then the command pool (which frees its command buffers
        // implicitly -- no separate free_command_buffers call needed).
        unsafe {
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_frame_wraps_around() {
        assert_eq!(next_frame(0), 1);
        assert_eq!(next_frame(FRAMES_IN_FLIGHT - 1), 0);
    }
}
