# 002: Command buffers, synchronization, and the render loop

**Roadmap item**: [ROADMAP.md](../ROADMAP.md) Phase 1 — "Command pool & buffers", "Synchronization", and
"First triangle" bullets.

**Depends on**: [001-graphics-pipeline.md](001-graphics-pipeline.md) — needs a working `GraphicsPipeline`.

**Learning path**: [LEARNING_PATH.md](../LEARNING_PATH.md) Phase 1 — command buffers and synchronization
sections. Primary source for this spec's shape:
[Vulkan Tutorial: Rendering and presentation](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Rendering_and_presentation)
and [Frames in flight](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Frames_in_flight) — **but read
the caveat below before following it line-for-line.**

## The dynamic rendering gap in the tutorial

The tutorial's render loop assumes a `VkRenderPass`. A render pass's subpass dependencies do the
`UNDEFINED → COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR` image layout transitions *for you*, implicitly, as
part of `vkCmdBeginRenderPass`/`vkCmdEndRenderPass`. Spec 001 committed this codebase to
`VK_KHR_dynamic_rendering` instead, which has no subpass dependencies — **the layout transitions have to be
recorded explicitly**, as `vkCmdPipelineBarrier` calls, or the validation layer will reject the draw (and
without validation, you'd just get a black window or garbage). This is the one place this spec cannot be a
straight tutorial port. Reference for the barrier shape:
[Vulkan-Samples: dynamic_rendering](https://docs.vulkan.org/samples/latest/samples/extensions/dynamic_rendering/README.html)
(already linked from `LEARNING_PATH.md`) shows both barriers.

## Goal

Implement `command.rs` as a `FrameContext` that owns a command pool, one command buffer and one set of
sync primitives per frame-in-flight, and a `draw_frame` method that runs the full per-frame protocol: wait,
acquire, record, submit, present. Wire it into `VulkanRenderer::render_frame()` so `cargo run --example
triangle` actually shows the hardcoded triangle from spec 001's shaders.

## Non-goals

- **Swapchain recreation on resize.** `acquire_next_image`/`queue_present` returning
  `ERROR_OUT_OF_DATE_KHR` is handled by logging and skipping the frame, not rebuilding the swapchain. Resizing
  the window during this milestone will eventually stop rendering (or the app may need a restart) until a
  later spec addresses this roadmap item. Flagging loudly rather than silently swallowing it.
- **Real vertex buffers.** Still hardcoded positions from spec 001's shaders — `003-vertex-buffers.md`.
- **Per-swapchain-image (rather than per-frame-in-flight) semaphores.** The classic tutorial pattern uses
  `FRAMES_IN_FLIGHT` semaphores even though there can be a different number of swapchain images, which has a
  known edge case (a semaphore can theoretically be reused while still in use if present and acquire timing
  line up badly on some drivers/present modes). This is the same simplification the tutorial makes; noting it
  as a known limitation rather than solving it now.

## Public API

```rust
// src/renderer/command.rs

pub const FRAMES_IN_FLIGHT: usize = 2;

pub struct FrameContext {
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    image_available_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    render_finished_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    current_frame: usize,
    device: ash::Device, // same pattern as GraphicsPipeline: cloned handle, for Drop only
}

impl FrameContext {
    pub fn new(device: &VulkanDevice) -> Result<Self>;

    /// Runs one full frame: waits on this frame-in-flight's fence, acquires a swapchain image,
    /// records the hardcoded-triangle draw against it, submits, and presents. Advances the
    /// internal frame-in-flight counter on success. On `ERROR_OUT_OF_DATE_KHR`/`SUBOPTIMAL_KHR`
    /// from either acquire or present, logs a warning and returns `Ok(())` without drawing (see
    /// Non-goals) rather than propagating an error the caller has no way to act on yet.
    pub fn draw_frame(
        &mut self,
        device: &VulkanDevice,
        pipeline: &GraphicsPipeline,
        swapchain: &VulkanSwapchain,
    ) -> Result<()>;
}

impl Drop for FrameContext {
    fn drop(&mut self); // destroy fences, semaphores, then command_pool (frees its buffers)
}
```

`VulkanRenderer` (`mod.rs`) gains two new fields — `pipeline: ManuallyDrop<pipeline::GraphicsPipeline>` and
`frame_context: ManuallyDrop<command::FrameContext>` — constructed after `swapchain` in `new`, and
`render_frame()` becomes:

```rust
pub fn render_frame(&mut self) -> Result<()> {
    self.frame_context.draw_frame(&self.device, &self.pipeline, &self.swapchain)
}
```

## Vulkan objects

| Object | Count | Created by | Notes |
|---|---|---|---|
| `vk::CommandPool` | 1 | `FrameContext::new` | `vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER` — lets us reset individual command buffers per-frame instead of resetting the whole pool |
| `vk::CommandBuffer` | `FRAMES_IN_FLIGHT` | Allocated from the pool, `LEVEL_PRIMARY` | One per frame-in-flight, reused (reset + re-recorded) every frame it comes back around to |
| `vk::Semaphore` (image-available) | `FRAMES_IN_FLIGHT` | `FrameContext::new` | GPU-GPU: signals when the acquired image is ready to be written to |
| `vk::Semaphore` (render-finished) | `FRAMES_IN_FLIGHT` | `FrameContext::new` | GPU-GPU: signals when rendering is done, gates the present |
| `vk::Fence` (in-flight) | `FRAMES_IN_FLIGHT` | `FrameContext::new`, **created signaled** (`SIGNALED` flag) | CPU-GPU: lets the CPU know when it's safe to reuse this frame-in-flight's command buffer. Signaled-at-creation avoids a permanent stall on the very first `wait_for_fences` call, before any frame has ever been submitted |

**Destroy order** (extends the `ManuallyDrop` pattern from `mod.rs`'s existing `Drop`, see the fix already
applied there): fences, then semaphores, then command pool (which frees its command buffers implicitly — no
separate `free_command_buffers` call needed). `FrameContext` itself must be destroyed before `VulkanDevice`,
same constraint as `GraphicsPipeline`. In `VulkanRenderer`'s field order, both `pipeline` and `frame_context`
are constructed after `swapchain`, so both must be torn down *before* `swapchain`, `device`, and `instance` in
the explicit `Drop` — reverse of construction order, same rule as before, one more link in a now four-deep
chain: `frame_context → pipeline → swapchain → device → instance`.

## Interactions

- `mod.rs`: `VulkanRenderer` struct (new fields), `new()` (construct after swapchain), `render_frame()`
  (real body instead of a no-op), `Drop` (extend the explicit teardown order — see above).
- `pipeline.rs`: `GraphicsPipeline` needs to be readable (pipeline handle, layout) by `command.rs`'s recording
  code — likely `pub(crate)` accessors or `pub(crate)` fields rather than fully private, since this is an
  internal cross-module call within the same crate, not a public API surface.
- `swapchain.rs`: none — `images`, `image_views`, `format`, `extent` are already `pub`.

## Recording sequence (inside `draw_frame`, per acquired image)

1. `vkCmdPipelineBarrier`: `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL`. `src_access_mask` empty, `dst_access_mask
   = COLOR_ATTACHMENT_WRITE`, `src_stage = TOP_OF_PIPE`, `dst_stage = COLOR_ATTACHMENT_OUTPUT`.
2. `vkCmdBeginRendering` with one color attachment: this image view, `layout =
   COLOR_ATTACHMENT_OPTIMAL`, `load_op = CLEAR` (clear color — anything but black makes it obvious the clear
   is actually happening, independent of whether the triangle draws), `store_op = STORE`.
3. `vkCmdBindPipeline(GRAPHICS, pipeline)`.
4. `vkCmdSetViewport` / `vkCmdSetScissor` — required every time this command buffer is recorded, since the
   pipeline declared them dynamic (spec 001) rather than baking them in.
5. `vkCmdDraw(3, 1, 0, 0)` — three hardcoded vertices, one instance.
6. `vkCmdEndRendering`.
7. `vkCmdPipelineBarrier`: `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR`. `src_access_mask =
   COLOR_ATTACHMENT_WRITE`, `dst_access_mask` empty, `src_stage = COLOR_ATTACHMENT_OUTPUT`, `dst_stage =
   BOTTOM_OF_PIPE`.

## Per-frame protocol (inside `draw_frame`, around the recording above)

1. `wait_for_fences([in_flight_fences[current_frame]], true, u64::MAX)`.
2. `acquire_next_image(swapchain, u64::MAX, image_available_semaphores[current_frame], vk::Fence::null())`.
   On `ERROR_OUT_OF_DATE_KHR`: log + return `Ok(())` (see Non-goals) *before* resetting the fence — if we
   bail here, the fence must stay signaled from step 1, or the next call's wait blocks forever.
3. `reset_fences([in_flight_fences[current_frame]])` — only once we know we're actually submitting.
4. `reset_command_buffer` + record (sequence above), targeting the acquired image index's `vk::Image`/
   `vk::ImageView`.
5. `queue_submit`: wait on `image_available_semaphores[current_frame]` at stage
   `COLOR_ATTACHMENT_OUTPUT`, signal `render_finished_semaphores[current_frame]`, fence =
   `in_flight_fences[current_frame]`.
6. `queue_present`: wait on `render_finished_semaphores[current_frame]`. On `ERROR_OUT_OF_DATE_KHR` /
   `SUBOPTIMAL_KHR`: log + continue (see Non-goals).
7. `current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT`.

## Test plan

- Unit tests: not much pure logic here either (see spec 001's same note) — `(current_frame + 1) %
  FRAMES_IN_FLIGHT` wraparound is the one pure-logic piece worth a table test if it ends up factored into its
  own function.
- Manual verification (this spec's actual milestone): `cargo run --example triangle` shows a colored triangle
  on the clear-colored background. Close the window; confirm the shutdown log is still clean (no validation
  errors) — the four-deep `ManuallyDrop` teardown chain is new territory relative to what was verified after
  the earlier `mod.rs` fix, so this is the first real test of it with two more object types in the chain.

## Open questions

- `GraphicsPipeline` fields are currently private (spec 001) — decide `pub(crate)` fields vs. accessor
  methods when `command.rs` needs to read the pipeline/layout handles. Either is fine; pick one and be
  consistent with whatever `FrameContext` itself exposes back to `mod.rs`.

## Outcome

_(Filled in after implementation.)_
