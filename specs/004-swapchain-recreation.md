# 004: Swapchain recreation on resize

**Roadmap item**: [ROADMAP.md](../ROADMAP.md) Phase 1 — "Swapchain recreation on resize".

**Depends on**: [001-graphics-pipeline.md](001-graphics-pipeline.md) (dynamic viewport/scissor + dynamic
rendering — why only the swapchain needs recreating) and
[002-command-buffers-and-render-loop.md](002-command-buffers-and-render-loop.md) (the render loop +
`ERROR_OUT_OF_DATE_KHR` handling this replaces).

**Learning path**: Vulkan Tutorial's
[Swap chain recreation](https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation) chapter —
**adapted**: there are no framebuffers or render pass to recreate here (dynamic rendering), and the pipeline is
size-independent (dynamic viewport/scissor), so this spec recreates *only* the swapchain + image views.

## Goal

Recreate the swapchain and its image views when the window resizes, so the quad keeps rendering correctly at
the new size instead of `002`'s log-and-skip. Driven by a caller-supplied `VulkanRenderer::resize(new_extent)`
notification, with `ERROR_OUT_OF_DATE_KHR` / `SUBOPTIMAL_KHR` from acquire/present as a backstop.

## Non-goals

- **Recreating the pipeline, framebuffers, or render pass.** Not needed — dynamic viewport/scissor (001) make
  the pipeline size-independent, and dynamic rendering (001) means no framebuffer/render-pass objects exist.
  Only the swapchain + image views are recreated. (This is the payoff of 001's decisions.)
- **Per-swapchain-image semaphores.** The `render_finished` reuse edge case (002 non-goal) is unchanged;
  recreation uses `device_wait_idle`, which sidesteps the common case but not that subtlety — still a later fix.
- **Reacting to format / color-space changes** (e.g. moving to an HDR monitor). Only the extent is handled;
  the surface format is assumed stable (as 001 assumed).
- **Resize-during-drag pipelining** beyond "recreate once per frame with the latest size" + passing the old
  swapchain as `oldSwapchain`.

## Design

### Detection & orchestration (caller-notified + backstop)

The library is window-agnostic (the caller owns the window and passed the initial extent to `new`), so a
resize is reported *to* the renderer:

- `VulkanRenderer::resize(new_extent)` records the new size as **pending** (it does not recreate immediately —
  one recreation per frame with the latest size, avoiding a storm during a drag).
- `render_frame` recreates **before drawing** if a recreation is pending, then draws.
- As a backstop, if `draw_frame` reports the swapchain went out of date (acquire/present returned
  `ERROR_OUT_OF_DATE_KHR` / `SUBOPTIMAL_KHR`), `render_frame` marks a recreation pending for the next frame.

`recreate` re-queries the surface capabilities, so on platforms where `currentExtent` is authoritative
(macOS/MoltenVK/kosmickrisp) it uses the true new size regardless; the caller-supplied extent is the fallback
when `currentExtent == u32::MAX` (handled by the existing `choose_swap_extent`).

### Minimization

A minimized window reports a `0 × 0` extent, which can't make a swapchain. When the pending extent has a zero
dimension, `render_frame` **skips the frame** (keeps the pending flag) and resumes once it's non-zero again.

## Public API

```rust
// src/renderer/mod.rs
impl VulkanRenderer {
    /// Notify the renderer that the window surface resized to `new_extent` (physical pixels). Stored
    /// as pending; the next `render_frame` recreates the swapchain before drawing. A zero dimension
    /// (minimized) pauses rendering until it's non-zero again.
    pub fn resize(&mut self, new_extent: (u32, u32));
}

// src/renderer/swapchain.rs
impl VulkanSwapchain {
    /// Recreate the swapchain + image views at `new_extent`, reusing the existing surface. The caller
    /// must have waited for the device to be idle. Destroys the old image views + swapchain first.
    pub(crate) fn recreate(&mut self, device: &VulkanDevice, new_extent: (u32, u32)) -> Result<()>;
}

// src/renderer/command.rs
/// Whether a frame drew or the swapchain needs recreating.
pub enum FrameOutcome {
    Rendered,
    SwapchainOutOfDate,
}
// draw_frame(...) -> Result<FrameOutcome>   (was Result<()>, logging-and-skipping on out-of-date)
```

## Vulkan objects

| Object | On recreate |
|---|---|
| `vk::SwapchainKHR` | old destroyed; new created at the new extent, passing the old as `old_swapchain` |
| `vk::ImageView` (swapchain images) | old destroyed; new created for the new images |
| `vk::SurfaceKHR` | **reused** — never recreated |
| pipeline / framebuffers / render pass | untouched — size-independent / nonexistent |

`recreate` destroys the old image views (via the device) then the old swapchain (via the swapchain loader),
before building the new ones. The **caller runs `device_wait_idle` first** so nothing is mid-use. `Drop` is
unchanged (still destroys the final swapchain + surface).

DRY: `new` currently inlines "create swapchain, get images, create image views." Factor that into a private
helper (e.g. `build(device, new_extent, old_swapchain)`) that both `new` and `recreate` call, rather than
duplicating it (CLAUDE.md "shared before one-off").

## Interactions

- **`swapchain.rs`:** add `recreate`; extract the shared build helper. `format` stays constant across recreation.
- **`mod.rs`:** `VulkanRenderer` gains `resize` + a `pending_resize: Option<(u32, u32)>` field; `render_frame`
  recreates before drawing and reacts to `draw_frame`'s `FrameOutcome`; skips when the pending extent is zero.
- **`command.rs`:** `draw_frame` returns `FrameOutcome` instead of logging-and-returning-`Ok` on out-of-date.
- **`examples/triangle.rs`:** the `WindowEvent::Resized` handler (currently just logs) calls
  `renderer.resize((size.width, size.height))`.
- **`error.rs`:** none.

## Test plan

- **Unit tests** (pure logic): `choose_swap_extent` is already covered. Add a small `is_renderable`/zero-extent
  check if it ends up factored into its own function (the one pure branch worth a test).
- **Manual verification** (the milestone): run the example and resize the window — drag a corner, maximize,
  restore. The quad stays correctly proportioned and keeps rendering (no stretching, no freeze). Minimize then
  restore: rendering pauses then resumes. No validation errors during any of it; clean teardown on close.

## Open questions (resolved)

- **`SUBOPTIMAL_KHR` on present**: **recreate on it too**, same as `OUT_OF_DATE` — keeps the image crisp
  instead of waiting for a hard out-of-date.
- **Semaphore state across recreation**: **deferred.** The `wait_idle` + recreate path is correct for the
  resize itself. The per-image-`render_finished` fix (002 carryover) is a steady-state sync-correctness issue,
  orthogonal to resize, and a *validation-only* bug we can't verify while validation layers are off. It lands
  after **005 (enable validation layers)**, where it can actually be confirmed.
- **Pending-resize representation**: **`pending_resize: Option<(u32, u32)>`** — `Some` = recreate pending at
  that size, `.take()` to recreate; `OUT_OF_DATE`/`SUBOPTIMAL` set `Some(current swapchain extent)` as a
  fallback (fine, since `recreate` re-queries surface caps anyway).

## Outcome

_(Filled in after implementation.)_
