// Command pool/buffer management, synchronization, and the render loop. See
// specs/002-command-buffers-and-render-loop.md for the full design -- especially the "dynamic
// rendering gap" section, since this is the one part of the render loop that can't be a
// straight port of vulkan-tutorial.com's render-pass-based version.

use crate::error::Result;
use crate::renderer::device::VulkanDevice;
use crate::renderer::pipeline::GraphicsPipeline;
use crate::renderer::swapchain::VulkanSwapchain;
use ash::vk;

pub const FRAMES_IN_FLIGHT: usize = 2;

// `command_buffers`/`current_frame` aren't read yet because `draw_frame` below is still a
// `todo!()` -- remove this once its real body (which indexes both by `current_frame`) exists.
#[allow(dead_code)]
pub struct FrameContext {
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    image_available_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    render_finished_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    current_frame: usize,
    device: ash::Device, // cloned handle, kept only so Drop can destroy the fields above
}

impl FrameContext {
    pub fn new(_device: &VulkanDevice) -> Result<Self> {
        // TODO(spec 002, step 1): create the command pool.
        // `vk::CommandPoolCreateInfo::default().queue_family_index(device.graphics_queue_family)
        //     .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)` -- the RESET flag matters:
        // it's what lets step 2 reset individual command buffers per-frame instead of having to
        // reset the whole pool. Source: https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
        // ("Command pools" section) -- this part IS a straight port, no dynamic-rendering wrinkle.

        // TODO(spec 002, step 2): allocate FRAMES_IN_FLIGHT primary command buffers from the pool
        // in one `allocate_command_buffers` call (`level(vk::CommandBufferLevel::PRIMARY)`,
        // `command_buffer_count(FRAMES_IN_FLIGHT as u32)`), then `.try_into().unwrap()` the
        // returned `Vec` into `[vk::CommandBuffer; FRAMES_IN_FLIGHT]`.

        // TODO(spec 002, step 3): create the sync objects, FRAMES_IN_FLIGHT of each. Semaphores:
        // `vk::SemaphoreCreateInfo::default()`, nothing else needed. Fences: MUST pass
        // `vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED)` -- read the
        // "Creating the synchronization objects" section of
        // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Rendering_and_presentation
        // for exactly why signaled-at-creation matters (what happens on the very first
        // `wait_for_fences` call if you get this wrong).

        todo!("implement per specs/002-command-buffers-and-render-loop.md -- steps 1-3 above, in order")
    }

    /// Runs one full frame against `swapchain`, using `pipeline` to draw. See
    /// specs/002-command-buffers-and-render-loop.md's "Per-frame protocol" and "Recording
    /// sequence" sections -- this method is both interleaved together in the real implementation.
    pub fn draw_frame(
        &mut self,
        _device: &VulkanDevice,
        _pipeline: &GraphicsPipeline,
        _swapchain: &VulkanSwapchain,
    ) -> Result<()> {
        // TODO(spec 002, step 4): wait_for_fences([in_flight_fences[current_frame]], true, u64::MAX).

        // TODO(spec 002, step 5): acquire_next_image(swapchain.swapchain, u64::MAX,
        // image_available_semaphores[current_frame], vk::Fence::null()). On
        // vk::Result::ERROR_OUT_OF_DATE_KHR specifically: log::warn! and `return Ok(())`
        // immediately -- BEFORE resetting the fence in the next step, or the fence you just
        // waited on above stays unsignaled forever and the next call deadlocks. (See Non-goals
        // in the spec: swapchain recreation is a later spec, not this one.)

        // TODO(spec 002, step 6): reset_fences([in_flight_fences[current_frame]]) -- only once
        // you know you're actually about to submit (i.e. after the out-of-date check above, not
        // before it).

        // TODO(spec 002, step 7): reset_command_buffer(command_buffers[current_frame], ...) then
        // record it:
        //   a. vkCmdPipelineBarrier: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL on the acquired
        //      image (swapchain.images[image_index]). See the spec's "dynamic rendering gap"
        //      section and the Vulkan-Samples link there for the exact barrier fields.
        //   b. vkCmdBeginRendering: one color attachment using
        //      swapchain.image_views[image_index], layout COLOR_ATTACHMENT_OPTIMAL, LOAD_OP_CLEAR
        //      with a clear color, STORE_OP_STORE. render_area = swapchain.extent.
        //   c. vkCmdBindPipeline(GRAPHICS, pipeline.pipeline).
        //   d. vkCmdSetViewport + vkCmdSetScissor, sized to swapchain.extent -- required every
        //      recording since the pipeline declared these dynamic (spec 001).
        //   e. vkCmdDraw(3, 1, 0, 0).
        //   f. vkCmdEndRendering.
        //   g. vkCmdPipelineBarrier: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR (reverse of a.).

        // TODO(spec 002, step 8): queue_submit on device.graphics_queue: wait on
        // image_available_semaphores[current_frame] at stage COLOR_ATTACHMENT_OUTPUT, signal
        // render_finished_semaphores[current_frame], fence = in_flight_fences[current_frame].

        // TODO(spec 002, step 9): queue_present on device.present_queue: wait on
        // render_finished_semaphores[current_frame]. On ERROR_OUT_OF_DATE_KHR/SUBOPTIMAL_KHR:
        // log::warn! and continue rather than propagating an error (see Non-goals).

        // TODO(spec 002, step 10): current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT.

        todo!("implement per specs/002-command-buffers-and-render-loop.md's per-frame protocol")
    }
}

impl Drop for FrameContext {
    fn drop(&mut self) {
        // Fences, then semaphores, then the command pool (which frees its command buffers
        // implicitly -- no separate free_command_buffers call needed). This part isn't a TODO --
        // it's already correct and will stay correct once `new()` above populates real handles.
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
