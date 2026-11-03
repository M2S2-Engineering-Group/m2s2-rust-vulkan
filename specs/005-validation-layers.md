# 005: Validation layers (and fixing what they flag)

**Roadmap item**: [ROADMAP.md](../ROADMAP.md) Phase 0 foundation — "validation layers" (listed as foundational,
but the layer was never actually *enabled*: `instance.rs` has a debug messenger yet an empty `layer_names`).
Not the Phase-4 "Debug tooling integration" item (object naming / RenderDoc).

**Depends on**: everything so far — this turns the safety net on over the whole renderer.

**Learning path**: [LEARNING_PATH.md](../LEARNING_PATH.md) Phase 0. Vulkan Tutorial's
[Validation layers](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers) chapter.

## Goal

Actually enable `VK_LAYER_KHRONOS_validation` in debug builds (gated on availability so release builds and
CI-without-the-SDK still work), run the engine under it, and **fix every validation error it reports** until the
example runs clean through render + resize + minimize + close. This removes the "validation is off, so 'no
validation errors' is only as strong as the debug messenger" caveat carried since 001, and is the verified home
for the fixes deferred from 002/004.

## Non-goals

- **Release-build validation.** Debug-only, matching the existing `cfg(debug_assertions)` messenger gating.
- **GPU-assisted validation, best-practices, or synchronization2 layers.** Just the core validation layer.
- **A configurable/env-driven toggle** beyond `debug_assertions`. Keep it simple; revisit if needed.
- **Performance-warning triage.** Fix `ERROR`s and correctness-relevant `WARNING`s; perf hints are out of scope.
- **`VK_EXT_debug_utils` object naming / labels.** That's the separate Phase-4 debug-tooling item.

## Decision: debug-only, availability-gated

Unconditionally requesting the layer fails `create_instance` with `ERROR_LAYER_NOT_PRESENT` when it's absent —
which is exactly CI (`ubuntu-latest` has no Vulkan SDK) and any machine without the SDK. So: enumerate
`enumerate_instance_layer_properties`, enable the layer **only if present**, otherwise log a warning and carry
on (the debug messenger still attaches; it just receives no validation messages). This keeps CI green and the
engine robust. Verified locally available: `vulkaninfo` lists `VK_LAYER_KHRONOS_validation` 1.4.350.

## Design

### Enable the layer (`instance.rs`)

- `const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";`
- A helper that returns whether the layer is available (scans `enumerate_instance_layer_properties`).
- In `cfg(debug_assertions)`: if available, push `VALIDATION_LAYER.as_ptr()` into `layer_names`; else
  `log::warn!` that validation was requested but is unavailable.
- **Also chain a `DebugUtilsMessengerCreateInfoEXT` into `InstanceCreateInfo`'s `p_next`** so validation
  messages emitted *during* `create_instance`/`destroy_instance` are caught too (the persistent messenger only
  covers the window in between). Standard LunarG setup.

### Fix everything validation flags

Enable it, run the example (render, resize, minimize, close), and fix each reported error. Known candidates
expected to surface (carried over), with their intended fixes:

- **Shutdown image-view leak** (`swapchain.rs`): `Drop` destroys the swapchain + surface but not the image
  views — the struct holds no device handle. Fix: store a cloned `ash::Device` on `VulkanSwapchain` and destroy
  the image views in `Drop` (before the swapchain). Follows the pattern every other wrapper uses. `recreate`
  already destroys the *old* views each resize; this closes the final-teardown case.
- **Per-image `render_finished` semaphores** (`command.rs`): `002` made `render_finished` per
  *frame-in-flight* (2), but present waits on it with no fence to say when it's consumed, so with more swapchain
  images than `FRAMES_IN_FLIGHT` a semaphore can be reused while still pending. Fix: make `render_finished` one
  **per swapchain image**, indexed by the acquired image index; `image_available` stays per-frame-in-flight.
  The per-image semaphores are sized to the swapchain image count at `FrameContext::new` and **recreated when
  the image count changes** on swapchain recreation (a `pub(crate)` method the renderer calls from
  `render_frame` right after `swapchain.recreate`).
- **Anything else validation reports** — fix as found; the acceptance criterion is zero validation errors, so
  the exact set is whatever the first validated runs surface.

## Public API

Mostly internal:

- `instance.rs`: no public API change (layer enabling is internal).
- `swapchain.rs`: `VulkanSwapchain` gains a private `device: ash::Device` field; `Drop` destroys image views.
- `command.rs`: `FrameContext` internals change (`render_finished` → per-image `Vec`); `FrameContext::new`
  needs the swapchain image count (takes `&VulkanSwapchain` or a count); add a `pub(crate)` method to recreate
  the per-image semaphores when the image count changes.
- `mod.rs`: after `swapchain.recreate`, call `frame_context`'s semaphore-recreate; pass the image count to
  `FrameContext::new`.

## Vulkan objects

| Object | Change |
|---|---|
| validation layer | enabled at instance creation (debug + available) |
| instance `p_next` messenger | added so create/destroy-instance messages are caught |
| swapchain image views | now destroyed in `VulkanSwapchain::Drop` (leak fix) |
| `render_finished` semaphores | re-sized to **one per swapchain image**; recreated on swapchain recreate |

## Interactions

- `instance.rs` (enable layer + p_next messenger), `swapchain.rs` (device field + Drop), `command.rs`
  (`FrameContext` per-image semaphores + recreate method + `new` takes image count), `mod.rs` (pass image count;
  drive semaphore recreate alongside swapchain recreate). `error.rs`: none expected.

## Test plan

- **Unit tests**: no new pure logic beyond what exists, unless a helper falls out (e.g. the frame↔image index
  mapping). The layer check and semaphore wiring are device-dependent.
- **Manual verification** (the milestone): `cargo run --example triangle` in a debug build logs that validation
  is **active** and runs with **zero validation errors/warnings** through: steady rendering, drag-resize,
  maximize/restore, minimize/restore, and window close (teardown). A debug build with the SDK present is the
  real gate; also confirm a build *without* the layer available degrades gracefully (warn + run).

## Open questions

- If a validation error surfaces that's genuinely in another module's territory and large, is it in-scope or a
  fast-follow? Default: fix it here (the point is a clean run), unless it balloons — then note + defer.
- Does kosmickrisp's validation flag the per-image-semaphore issue, or only desktop loaders? Either way the fix
  is correct; implementation confirms whether it was actively triggering.

## Outcome

Implemented and verified — **validation enabled and the engine runs clean**: confirmed active in the log
(loader inserts `VK_LAYER_KHRONOS_validation` at instance + device level), with **zero validation
errors/warnings** through render, resize-recreate, and **graceful teardown** on an Apple M5 / `kosmickrisp`
driver. The whole engine (specs 001–004) is now actually validated, not just assumed.

Enable path: debug-only, availability-gated (`enumerate_instance_layer_properties`; warn-and-skip if absent, so
CI/SDK-less builds are fine), plus a `p_next` messenger for create/destroy-instance messages.

Validation flagged **three** issues, all fixed:
1. **`VK_KHR_dynamic_rendering` missing prerequisites** (a surprise beyond the two known debts): on a 1.0-API
   instance it needs `VK_KHR_get_physical_device_properties2` (instance) + the `depth_stencil_resolve →
   create_renderpass2 → {multiview, maintenance2}` device chain. Added them. *(This is the real cost of spec
   001's "extension on a 1.0 instance" choice; bumping the API version to 1.2/1.3 would drop this dependency
   tail — a possible future 001 simplification, not done here.)*
2. **Per-frame `render_finished` semaphore reuse** (the predicted 002 debt — and it *did* trigger on
   kosmickrisp): made `render_finished` one per swapchain image, recreated on swapchain recreate.
3. **Shutdown image-view leak** (the 004 find): `VulkanSwapchain` now holds a cloned `ash::Device` and destroys
   its image views in `Drop`.

No further errors surfaced beyond these three; the example is clean across all exercised paths.
