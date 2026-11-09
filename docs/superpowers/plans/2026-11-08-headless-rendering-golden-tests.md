# Headless Rendering and Golden-Image Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the crate a test-only, `#[cfg(test)]`-gated way to render a scene with no window, no swapchain, and no display server, read the pixels back to the CPU, and compare them against a committed reference PNG within a small tolerance — general-purpose infrastructure any future rendering feature's tests can reuse, starting with spec 018 (point lighting).

**Architecture:** A headless `VulkanInstance`/`VulkanDevice` pair that skips every surface/present/swapchain requirement, feeding a small independent `HeadlessRenderer` (not a mode of `VulkanRenderer` — that struct's swapchain coupling runs too deep to safely branch). `HeadlessRenderer` owns an `OffscreenTarget` (a plain color `VkImage` standing in for a swapchain image) and reuses the existing `emit_object_draws`/pipeline/depth/uniform machinery verbatim. A pure `golden.rs` comparison module (PNG decode/encode + per-channel tolerance) is the only piece reachable from ordinary `cargo test --all-targets` without a GPU; everything else is `#[ignore]`d.

**Tech Stack:** Rust, `ash` (Vulkan 1.3), `gpu-allocator`, `bytemuck`; new dev-dependency `png` for golden-image encode/decode.

**Spec:** `specs/019-headless-rendering-and-golden-image-tests.md`

## Global Constraints

- **Nothing in this plan is public API.** Every new item is `pub(crate)`, and the `headless.rs`/`golden.rs` module tree is declared `#[cfg(test)]` at its `mod` declaration in `mod.rs` — none of it exists in a normal (non-test) build.
- **No window, no surface, ever.** `VulkanInstance::new_headless()` requests no `VK_KHR_surface`/platform-surface extension; `VulkanDevice::new_headless()` requires only a graphics-capable queue family, no present support, no `VK_KHR_swapchain`.
- Offscreen color format: **`R8G8B8A8_UNORM` preferred, `B8G8R8A8_UNORM` fallback** (queried via `get_physical_device_format_properties`, same pattern as `depth.rs`'s `select_depth_format` — no format asserted without a query). Readback byte order depends on which format was actually selected.
- `OffscreenTarget` is reusable across multiple `render_frame` calls on one instance: the very first call transitions `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL` with an empty source access mask from `TOP_OF_PIPE`; every call after the first transitions `TRANSFER_SRC_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL` with source access `TRANSFER_READ` from stage `TRANSFER` (the prior call's copy). A single `bool` ("has this target ever been rendered to") is sufficient to pick the right barrier — no general layout-tracking state machine is needed.
- `VulkanInstance::new_headless`'s debug messenger (debug builds only) counts `ERROR`-severity messages via a null-checked `pUserData` pointer to an `AtomicU32`; the production `with_window`/`unsafe new` path passes a null pointer and is unaffected.
- Golden comparison: per-channel absolute tolerance (0–255 scale), default **2**; `M2S2_UPDATE_GOLDEN` env var (any value set) writes the current render as the new golden instead of comparing.
- Golden PNGs live under `testdata/golden/` at the repo root (not `tests/` — that name is reserved for a separate integration-test crate in Rust and would be misleading here, since these tests live inside `src/renderer/` as `#[cfg(test)]` modules).
- Quality gates zero-warning at every commit, and **must pass without a GPU**: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test --all-targets`. GPU-dependent tests are `#[ignore]`d by default and run via `cargo test -- --ignored` — a command this plan's executor (no GPU available) **cannot run**; those tests are written and must compile/lint clean, but their pass/fail outcome is confirmed by the user, same as every prior spec's GPU acceptance step.
- Conventional Commits; **no `Co-Authored-By`/AI-attribution trailer**.

## Review Focus

- **`validation_error_count()` in a release build** (no validation layer, no debug messenger exists at all) → must still compile and return `0`, never panic or fail to exist. *Test in Task 1.*
- **`assert_matches_golden` against a golden PNG with different dimensions than `actual`** → a clear error, not a panic or silent out-of-bounds read. *Test in Task 4.*
- **Headless device selection finds zero physical devices with any graphics queue family** → a clean `InitializationError`, mirroring the existing `pick_physical_device`'s empty-device-list handling, not a panic or `unwrap`. *Test in Task 1 (structural — see Task 1 Step 7).*
- **The one-shot command submission fails partway through recording** (e.g. `end_command_buffer` fails after `begin_command_buffer` succeeded) → the transient command pool is still destroyed, mirroring `upload::submit_once`'s existing `Cleanup`-guard pattern; no leak. *Structural guarantee, not independently unit-testable without fault injection — Task 3 must follow the existing pattern exactly, reviewed by inspection.*
- **`OffscreenTarget`'s color-format fallback actually changes the readback byte order** (first candidate `R8G8B8A8_UNORM` needs no swizzle; the `B8G8R8A8_UNORM` fallback does) — if a test machine's driver only supports the fallback, a hardcoded "no swizzle" assumption would silently swap red and blue. *Test in Task 2 (pure, format-selection logic); the swizzle itself is exercised only by the GPU self-test in Task 5.*

---

### Task 1: Headless instance + device construction, with an assertable validation-error counter

Lands the two lowest-level pieces: a `VulkanInstance`/`VulkanDevice` pair that needs no window, surface, or display server, and the atomic error-counting mechanism the whole harness's "automated, not eyeballed" acceptance criterion depends on.

**Files:**
- Modify: `src/renderer/instance.rs` (`new_headless`, `validation_error_count`, thread `user_data` through `debug_messenger_create_info`/`setup_debug_messenger`, extend `debug_callback`)
- Modify: `src/renderer/device.rs` (`new_headless`, `find_graphics_queue_family`)

**Interfaces:**
- Consumes: existing `VulkanInstance::{entry, instance, debug_utils, debug_messenger}` fields; `VulkanDevice`'s existing `score_device_type`, `meets_min_api_version`, `MIN_DEVICE_API_VERSION`.
- Produces:
  - `pub(crate) fn VulkanInstance::new_headless() -> Result<Self>` (gated `#[cfg(test)]`)
  - `pub(crate) fn VulkanInstance::validation_error_count(&self) -> u32` (gated `#[cfg(test)]`; compiles and returns `0` in both debug and release)
  - `pub(crate) fn VulkanDevice::new_headless(instance: &VulkanInstance) -> Result<Self>` (gated `#[cfg(test)]`)

- [ ] **Step 1: Write the failing CPU test for the debug-callback counter**

In `src/renderer/instance.rs` tests (add a `#[cfg(test)] mod tests` if none exists — check first), write a test that calls `debug_callback` directly with a fabricated `vk::DebugUtilsMessengerCallbackDataEXT` (a real `CString` message, `p_message` pointing at it) and a real `AtomicU32` behind a raw pointer passed as `p_user_data`:

```rust
#[test]
fn debug_callback_increments_counter_only_on_error_severity() {
    use std::ffi::CString;
    use std::sync::atomic::{AtomicU32, Ordering};

    let counter = AtomicU32::new(0);
    let message = CString::new("test message").unwrap();
    let data = vk::DebugUtilsMessengerCallbackDataEXT::default().message(message.as_c_str());
    let user_data = &counter as *const AtomicU32 as *mut std::os::raw::c_void;

    unsafe {
        debug_callback(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            &data,
            user_data,
        );
    }
    assert_eq!(counter.load(Ordering::Relaxed), 0);

    unsafe {
        debug_callback(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            &data,
            user_data,
        );
    }
    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[test]
fn debug_callback_tolerates_null_user_data() {
    let message = std::ffi::CString::new("test message").unwrap();
    let data = vk::DebugUtilsMessengerCallbackDataEXT::default().message(message.as_c_str());
    unsafe {
        debug_callback(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            &data,
            std::ptr::null_mut(),
        );
    } // must not panic/segfault
}
```

(These tests only compile/run under `#[cfg(debug_assertions)]`, since `debug_callback` itself is gated that way — wrap both in the same `#[cfg(debug_assertions)]` the existing `debug_callback` uses, or accept that `cargo test` in a debug build is what CI/the pre-commit hook already runs.)

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib renderer::instance`
Expected: FAIL — `debug_callback` doesn't read `p_user_data` at all yet (counter stays 0 after the ERROR call too).

- [ ] **Step 3: Extend `debug_callback` to increment a null-checked counter**

In `src/renderer/instance.rs`, change `debug_callback`'s unused `_user_data: *mut std::os::raw::c_void` parameter to `user_data: *mut std::os::raw::c_void`, and inside the function, after logging: if `message_severity` is `ERROR` and `user_data` is non-null, treat it as `*const std::sync::atomic::AtomicU32` and `fetch_add(1, Ordering::Relaxed)`. Guard the pointer dereference with an explicit null check before casting — never dereference a null pointer.

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --lib renderer::instance`
Expected: PASS.

- [ ] **Step 5: Thread `user_data` through the debug-messenger setup, and add `new_headless` + `validation_error_count`**

In `src/renderer/instance.rs`:
- Change `debug_messenger_create_info` to take a `user_data: *mut std::os::raw::c_void` parameter and call `.user_data(user_data)` on the builder (existing call site in `setup_debug_messenger` passes `std::ptr::null_mut()`, preserving today's behavior).
- Change `setup_debug_messenger` to take and forward the same `user_data` parameter.
- Add a new `#[cfg(debug_assertions)]` field on `VulkanInstance`: `validation_errors: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>` (always `None` for the existing `new()`; `Some(...)` only for `new_headless()`).
- Add `VulkanInstance::new_headless() -> Result<Self>`: builds an `InstanceCreateInfo` requesting only `VK_EXT_debug_utils` (debug builds) and no surface extensions — reuse `entry`/`app_info`/layer-enabling logic from `new()` but skip the `extension_names.push(ash::khr::surface::NAME...)` and every platform-`#[cfg]` surface push. In debug builds, create an `Arc<AtomicU32>`, pass `Arc::as_ptr(&counter) as *mut _` as `user_data` to `setup_debug_messenger`, and store the `Arc` in `validation_errors`.
- Add `validation_error_count(&self) -> u32`: in debug builds, `self.validation_errors.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0)`; in release builds (where the field doesn't exist), a separate `#[cfg(not(debug_assertions))]` body that always returns `0`.
- Gate `new_headless` and `validation_error_count` both `#[cfg(test)]` (no caller outside the new test-only module tree).

- [ ] **Step 6: Run the full CPU gate suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test --all-targets`
Expected: all clean. (`new_headless`/`validation_error_count` exist only under `#[cfg(test)]`, so clippy sees them as used by the tests in Step 1/this task's own upcoming GPU test — no dead-code warning.)

- [ ] **Step 7: Add `VulkanDevice::new_headless` + `find_graphics_queue_family`**

In `src/renderer/device.rs`, add (both `#[cfg(test)]`):
- `fn find_graphics_queue_family(instance: &VulkanInstance, physical_device: vk::PhysicalDevice) -> Option<u32>`: mirrors `find_queue_families`'s graphics-bit loop but with no surface/present check at all — returns the first queue family index with `vk::QueueFlags::GRAPHICS`.
- `fn pick_physical_device_headless(instance: &VulkanInstance) -> Result<(vk::PhysicalDevice, u32)>`: mirrors `pick_physical_device`'s enumerate/empty-check/score loop, but each candidate is checked only via `find_graphics_queue_family` (skip `supports_required_extensions` and `has_adequate_swapchain_support` entirely) plus `meets_min_api_version`. Same empty-device-list `InitializationError` as the existing function if `enumerate_physical_devices` returns nothing, and the same "no device qualifies" error if the loop finds none — this is the Review Focus item: mirror the defensive handling, don't `unwrap`.
- `VulkanDevice::new_headless(instance: &VulkanInstance) -> Result<Self>`: calls `pick_physical_device_headless`, builds a single `DeviceQueueCreateInfo` for the graphics family (no present-queue branch — there is only one queue family of interest), enables dynamic rendering via the same `push_next` pattern as `new`, passes an **empty** `enabled_extension_names` (no `VK_KHR_swapchain`). Sets `present_queue`/`present_queue_family` equal to `graphics_queue`/`graphics_queue_family` on the returned struct (there is nothing to present to; existing swapchain-based code never runs for a headless device).

- [ ] **Step 8: Run the full CPU gate suite again**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 9: Write the GPU construction test (cannot be run by this plan's executor — write it, confirm it compiles, leave pass/fail to the user)**

In `src/renderer/device.rs` tests (or a new `src/renderer/headless.rs` test module if Task 2 has already created that file — otherwise add here and move in Task 2):

```rust
#[test]
#[ignore = "needs a real Vulkan driver"]
fn headless_instance_and_device_construct_and_tear_down() {
    let instance = super::instance::VulkanInstance::new_headless().expect("headless instance");
    let device = VulkanDevice::new_headless(&instance).expect("headless device");
    assert_eq!(instance.validation_error_count(), 0);
    drop(device);
    drop(instance);
}
```

Run: `cargo build --all-targets` (confirms it compiles; do **not** attempt `cargo test -- --ignored` — no GPU in this environment).
Expected: builds clean. Note in the task report that this test's actual pass/fail is unverified here.

- [ ] **Step 10: Commit**

```bash
git add src/renderer/instance.rs src/renderer/device.rs
git commit -m "feat(renderer): headless instance/device construction with an assertable validation-error counter"
```

---

### Task 2: `OffscreenTarget` — a plain color render target with format selection and layout tracking

**Files:**
- Create: `src/renderer/headless.rs` (new module, declared `#[cfg(test)] pub(crate) mod headless;` in `mod.rs`)

**Interfaces:**
- Consumes: `buffer::Buffer::new_host_visible_mapped`, `Allocator`, `Cleanup` (spec 012's rollback pattern), `VulkanDevice`.
- Produces:
  - `pub(crate) const OFFSCREEN_COLOR_FORMAT_CANDIDATES: [vk::Format; 2]` = `[R8G8B8A8_UNORM, B8G8R8A8_UNORM]`.
  - `fn select_color_format(candidates: &[(vk::Format, vk::FormatFeatureFlags)]) -> Option<vk::Format>` (pure; mirrors `depth::select_depth_format` but checks `vk::FormatFeatureFlags::COLOR_ATTACHMENT`).
  - `pub(crate) struct OffscreenTarget { image, view, allocation: Option<Allocation>, readback: Buffer, extent: vk::Extent2D, format: vk::Format, has_rendered: bool }`
  - `OffscreenTarget::new(device: &VulkanDevice, instance: &VulkanInstance, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self>`
  - `OffscreenTarget::destroy(self, device: &VulkanDevice, allocator: &mut Allocator)`
  - `OffscreenTarget::format(&self) -> vk::Format`, `OffscreenTarget::has_rendered(&self) -> bool`, and a way for the one-shot recorder (Task 3) to mark it rendered (e.g. `pub(crate) fn mark_rendered(&mut self)`).

- [ ] **Step 1: Write the failing pure format-selection tests**

In `src/renderer/headless.rs`, create the file with a `#[cfg(test)] mod tests` containing (mirroring `depth.rs`'s three tests exactly, adapted to two candidates and `COLOR_ATTACHMENT`):

```rust
const COLOR_ATTACHMENT: vk::FormatFeatureFlags = vk::FormatFeatureFlags::COLOR_ATTACHMENT;

#[test]
fn prefers_first_candidate_that_supports_color_attachment() {
    let candidates = [
        (vk::Format::R8G8B8A8_UNORM, COLOR_ATTACHMENT),
        (vk::Format::B8G8R8A8_UNORM, COLOR_ATTACHMENT),
    ];
    assert_eq!(select_color_format(&candidates), Some(vk::Format::R8G8B8A8_UNORM));
}

#[test]
fn falls_back_when_first_candidate_unsupported() {
    let candidates = [
        (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::empty()),
        (vk::Format::B8G8R8A8_UNORM, COLOR_ATTACHMENT),
    ];
    assert_eq!(select_color_format(&candidates), Some(vk::Format::B8G8R8A8_UNORM));
}

#[test]
fn returns_none_when_no_candidate_supports_color_attachment() {
    let candidates = [
        (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::empty()),
        (vk::Format::B8G8R8A8_UNORM, vk::FormatFeatureFlags::empty()),
    ];
    assert_eq!(select_color_format(&candidates), None);
}
```

Add `#[cfg(test)] pub(crate) mod headless;` to `src/renderer/mod.rs`'s module list (alphabetical position, next to `pub mod depth;` or similar — match the file's existing ordering convention).

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib renderer::headless`
Expected: FAIL — `select_color_format` doesn't exist.

- [ ] **Step 3: Implement `select_color_format` and `OffscreenTarget`**

Above the test module in `headless.rs`, implement `select_color_format` (identical shape to `depth::select_depth_format`) and `OffscreenTarget`. `new` queries `get_physical_device_format_properties` for each of `OFFSCREEN_COLOR_FORMAT_CANDIDATES` (same pattern as `DepthResources::new`'s candidate-querying loop), picks via `select_color_format`, and on `None` returns `VulkanError::InitializationError("no supported offscreen color-attachment format")`. On a selected format, creates the `vk::Image` (`COLOR_ATTACHMENT | TRANSFER_SRC`, `OPTIMAL` tiling, `SAMPLE_COUNT_1`, `extent.{width,height}`, depth 1, `UNDEFINED` initial layout), allocates (`GpuOnly`, same `AllocationCreateDesc` shape as `DepthResources`), binds, creates the view (`COLOR` aspect), and creates the readback `Buffer` via `Buffer::new_host_visible_mapped(device, allocator, extent.width as u64 * extent.height as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST)`. Roll back partial resources on any failure exactly as `DepthResources::new` does (image created but allocation fails → destroy image and return; view fails → destroy image, free allocation, return). `has_rendered` starts `false`. Also implement the remaining declared surface: `destroy(self, device, allocator)` (destroy view + image via `device`, free the allocation and the readback buffer via `allocator` — same shape as `DepthResources::destroy`), `format(&self) -> vk::Format`, `has_rendered(&self) -> bool`, `mark_rendered(&mut self)` (sets the field to `true`).

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --lib renderer::headless`
Expected: PASS.

- [ ] **Step 5: Write a GPU test exercising `OffscreenTarget`'s full surface, so nothing is dead code**

`destroy`/`format`/`has_rendered`/`mark_rendered` have no caller yet within this task alone — add one `#[ignore]`d GPU test in the same module that exercises all of them, so Task 2 is self-contained and introduces no dead code (rather than deferring any of this task's own produced surface to Task 3):

```rust
#[test]
#[ignore = "needs a real Vulkan driver"]
fn offscreen_target_constructs_tracks_render_state_and_tears_down() {
    let instance = super::super::instance::VulkanInstance::new_headless().expect("instance");
    let device = super::super::device::VulkanDevice::new_headless(&instance).expect("device");
    let mut allocator = super::super::allocator::Allocator::new(&instance, &device).expect("allocator");
    let mut target = OffscreenTarget::new(&device, &instance, &mut allocator, vk::Extent2D { width: 4, height: 4 })
        .expect("offscreen target");
    assert!(!target.has_rendered());
    assert!(matches!(target.format(), vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM));
    target.mark_rendered();
    assert!(target.has_rendered());
    target.destroy(&device, &mut allocator);
}
```

Run: `cargo build --all-targets` (confirms it compiles; do not attempt `cargo test -- --ignored`).

- [ ] **Step 6: Run the full CPU gate suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test --all-targets`
Expected: all clean — every item `OffscreenTarget` produces now has a caller (the GPU test above), so no dead-code warning.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/headless.rs src/renderer/mod.rs
git commit -m "feat(renderer): OffscreenTarget — headless color render target with format selection"
```

---

### Task 3: `HeadlessRenderer` — construction, one-shot recording, and the determinism self-test

The largest task: widens `command.rs`'s draw-emission visibility, builds the independent `HeadlessRenderer` (mirroring `VulkanRenderer::init`'s resource-construction order without any swapchain/window concerns), and adds the one-shot frame-recording path with the correct repeated-call barrier logic.

**Files:**
- Modify: `src/renderer/command.rs` (`emit_object_draws`, `ObjectCommand` → `pub(super)`; no behavior change)
- Modify: `src/renderer/headless.rs` (`HeadlessRenderer`, the one-shot recording function, `OffscreenTarget::destroy`/`mark_rendered` wiring from Task 2)

**Interfaces:**
- Consumes: `OffscreenTarget` (Task 2), `VulkanInstance::new_headless`/`VulkanDevice::new_headless` (Task 1), `depth::DepthResources::new` (unchanged), `uniform::Uniforms::new`, `texture::create_set_layout`/`texture::StoredTexture` (for the white default — mirrors `VulkanRenderer::init`), `light::Light::new`, `pipeline::GraphicsPipeline::new`, `buffer::Buffer::new_device_local`, `command::{emit_object_draws, ObjectCommand}` (now `pub(super)`), `upload`'s one-shot submission pattern (reference for structure, not a shared function — this path begins/ends dynamic rendering, which `submit_once` doesn't).
- Produces:
  - `pub(crate) struct HeadlessRenderer { device: VulkanDevice, instance: VulkanInstance, allocator: ManuallyDrop<Allocator>, vertex_buffer: Buffer, index_buffer: Buffer, depth: DepthResources, uniforms: Uniforms, texture_set_layout: vk::DescriptorSetLayout, white_texture: StoredTexture, light: Light, pipeline: GraphicsPipeline, target: OffscreenTarget }`
  - `HeadlessRenderer::new(extent: (u32, u32), vertices: &[Vertex], indices: &[Index]) -> Result<Self>`
  - `HeadlessRenderer::render_frame(&mut self, objects: &[Object], camera: uniform::Camera) -> Result<Vec<u8>>`
  - `impl Drop for HeadlessRenderer` (consuming teardown of every owned resource, same order discipline as `VulkanRenderer`'s `Drop`)

- [ ] **Step 1: Widen `emit_object_draws`/`ObjectCommand` visibility**

In `src/renderer/command.rs`, change `enum ObjectCommand<'a>` and `fn emit_object_draws` from private (module-default) to `pub(super)`. No other change to either.

- [ ] **Step 2: Run the full CPU gate suite to confirm this alone is a no-op**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean, identical test counts to before this step (pure visibility change).

- [ ] **Step 3: Write the failing GPU tests for construction and the determinism self-test**

In `src/renderer/headless.rs`'s test module (move the construction test from Task 1 Step 9 here if it was placed in `device.rs`, so all `HeadlessRenderer`-adjacent GPU tests live together):

```rust
fn lit_untextured_cube() -> (Vec<Vertex>, Vec<Index>) {
    // Reuse the same 24-vertex-per-face cube geometry as examples/cube.rs (one outward normal per
    // face, no texture coordinates that matter since every Object below uses texture: None).
    // ... (exact vertex/index construction — copy the face-building approach from examples/cube.rs,
    // inlined here since examples aren't a dependency this crate's tests can import from)
}

#[test]
#[ignore = "needs a real Vulkan driver"]
fn headless_renderer_constructs_and_tears_down() {
    let (vertices, indices) = lit_untextured_cube();
    let renderer = HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
    drop(renderer);
}

#[test]
#[ignore = "needs a real Vulkan driver"]
fn render_frame_is_deterministic_across_repeated_calls() {
    let (vertices, indices) = lit_untextured_cube();
    let mut renderer = HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
    let camera = uniform::Camera {
        view: crate::math::look_at(&Vec3::new(2.0, 2.0, 2.0), &Vec3::new(0.0, 0.0, 0.0), &Vec3::new(0.0, 1.0, 0.0)),
        proj: crate::math::perspective(crate::math::to_radians(45.0), 1.0, 0.1, 10.0),
        position: Vec3::new(2.0, 2.0, 2.0),
    };
    let objects = [Object { model: Mat4::IDENTITY, texture: None, specular: SpecularMaterial::default() }];
    let first = renderer.render_frame(&objects, camera).expect("first render");
    let second = renderer.render_frame(&objects, camera).expect("second render");
    assert_eq!(first, second, "identical scene rendered twice must produce byte-identical output");
    assert_eq!(renderer_validation_errors(&renderer), 0); // exact accessor name per how Task 1's counter is threaded through
}
```

(The exact camera/geometry values aren't load-bearing — any fixed, non-degenerate scene satisfies the test. `renderer_validation_errors` is a placeholder name — wire up however `HeadlessRenderer` exposes the underlying `VulkanInstance::validation_error_count()`, e.g. a thin forwarding method.)

- [ ] **Step 4: Run to confirm they fail to compile (the types don't exist yet)**

Run: `cargo build --all-targets`
Expected: FAIL — `HeadlessRenderer` doesn't exist.

- [ ] **Step 5: Implement `HeadlessRenderer::new`**

Mirror `VulkanRenderer::init`'s resource-construction order exactly, substituting headless constructors: `VulkanInstance::new_headless()` → `VulkanDevice::new_headless(&instance)` → `Allocator::new(&instance, &device)` → `Buffer::new_device_local` for vertex/index → `DepthResources::new(&instance, &device, &mut allocator, extent)` → `Uniforms::new(&device, &mut allocator)` → `texture::create_set_layout(&device)` + `StoredTexture::new(..., &[255,255,255,255], 1, 1)` (the white default — needed because `resolve_texture_set` falls back to it for every `texture: None` object) → `Light::new(&device, &mut allocator)` → `GraphicsPipeline::new(&device, target_format, depth.format(), &[uniforms.descriptor_set_layout(), texture_set_layout, light.descriptor_set_layout()])` → `OffscreenTarget::new(&device, &instance, &mut allocator, extent)`. Use the same `Option`-held-locals-with-rollback-on-error pattern `VulkanRenderer::init` uses (spec 012) for every fallible step, since this constructor has the identical "partial success must not leak" requirement.

- [ ] **Step 6: Implement the one-shot recording function and `render_frame`**

A private function (or inherent method) that: creates a transient command pool + buffer (mirror `upload::submit_once`'s pool/buffer/fence creation and `Cleanup` guards exactly), begins the command buffer, transitions `target`'s image to `COLOR_ATTACHMENT_OPTIMAL` — if `!target.has_rendered()`: `old_layout(UNDEFINED)`, `src_access_mask(empty())`, source stage `TOP_OF_PIPE`; else: `old_layout(TRANSFER_SRC_OPTIMAL)`, `src_access_mask(TRANSFER_READ)`, source stage `TRANSFER` — begins dynamic rendering against the color view + depth view, binds the pipeline and the camera (set 0)/light (set 2) descriptor sets for the single frame slot (no frames-in-flight — index 0 throughout, since `Uniforms`/`Light` are still `PerFrameUniform`-backed with `FRAMES_IN_FLIGHT` slots; just always write/bind slot 0), writes the camera/light uniforms via their existing `update(0, ...)` methods, calls `emit_object_draws(objects, index_buffer.len(), &[], white_set, |cmd| match cmd { ... })` (binding set 1 per `BindTexture`, pushing per `PushModel`, drawing per `DrawIndexed` — same match arms as `command.rs::record`'s existing closure), ends rendering, transitions the color image to `TRANSFER_SRC_OPTIMAL` (`dst_access_mask(TRANSFER_READ)`, dest stage `TRANSFER`), issues `vkCmdCopyImageToBuffer` into `target.readback`, ends the command buffer, submits with a fence, waits, maps the readback buffer, copies into a `Vec<u8>`, and — if `target.format()` is `B8G8R8A8_UNORM` — swizzles B↔R per pixel (no swizzle needed for `R8G8B8A8_UNORM`). Marks `target.mark_rendered()` after a successful copy. Destroys the transient pool (frees the buffer implicitly) before returning.

- [ ] **Step 7: Implement `Drop for HeadlessRenderer`**

Consuming teardown in reverse construction order: `pipeline` (via its own `Drop`, matching `VulkanRenderer`'s pattern of letting `GraphicsPipeline`'s `Drop` run), `light.destroy`, `white_texture.destroy`, `texture_set_layout` (`destroy_descriptor_set_layout`), `uniforms.destroy`, `depth.destroy`, `index_buffer.destroy`, `vertex_buffer.destroy`, `target.destroy`, then drop the `allocator` (`ManuallyDrop::drop`), then `device`, then `instance` — the same ordering discipline `VulkanRenderer::Drop` already follows (allocator before device, device before instance, everything allocator-backed before the allocator itself).

- [ ] **Step 8: Build (cannot run the `#[ignore]`d tests — no GPU here)**

Run: `cargo fmt --all && cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test --all-targets`
Expected: all clean; the two new `#[ignore]`d tests are listed but not executed by the default `cargo test` run (confirm via the test-count-vs-ignored-count in the output). Do not attempt `cargo test -- --ignored`.

- [ ] **Step 9: Commit**

```bash
git add src/renderer/command.rs src/renderer/headless.rs
git commit -m "feat(renderer): HeadlessRenderer — one-shot offscreen rendering with repeated-call barrier handling"
```

---

### Task 4: `golden.rs` — PNG-backed comparison with tolerance (fully CPU-testable)

The one piece of this whole feature that needs no GPU at all to prove correct.

**Files:**
- Create: `src/renderer/golden.rs` (new module, declared `#[cfg(test)] pub(crate) mod golden;` in `mod.rs`)
- Modify: `Cargo.toml` (new dev-dependency)

**Interfaces:**
- Consumes: the `png` crate (decode/encode), `std::env::var` (for `M2S2_UPDATE_GOLDEN`).
- Produces: `pub(crate) fn assert_matches_golden(actual: &[u8], width: u32, height: u32, path: &std::path::Path, tolerance: u8) -> std::result::Result<(), String>`.

- [ ] **Step 1: Add the `png` dev-dependency**

In `Cargo.toml`, add `png = "0.17"` under `[dev-dependencies]`.

- [ ] **Step 2: Write the failing tests**

In `src/renderer/golden.rs`, create the file with a test module. Use `tempfile`-free temp paths via `std::env::temp_dir()` joined with a unique-per-test filename (no new dependency needed):

```rust
fn write_png(path: &std::path::Path, pixels: &[u8], width: u32, height: u32) {
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixels).unwrap();
}

#[test]
fn identical_images_pass_at_zero_tolerance() {
    let path = std::env::temp_dir().join("m2s2_golden_identical.png");
    let pixels = vec![10u8, 20, 30, 255, 40, 50, 60, 255]; // 2x1 RGBA
    write_png(&path, &pixels, 2, 1);
    assert!(assert_matches_golden(&pixels, 2, 1, &path, 0).is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn out_of_tolerance_pixel_fails_with_coordinates() {
    let path = std::env::temp_dir().join("m2s2_golden_mismatch.png");
    let golden = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
    write_png(&path, &golden, 2, 1);
    let mut actual = golden.clone();
    actual[4] = 200; // second pixel's red channel, far outside any reasonable tolerance
    let err = assert_matches_golden(&actual, 2, 1, &path, 2).unwrap_err();
    assert!(err.contains("1, 0") || err.contains("(1,0)"), "error should name the differing pixel's coordinates: {err}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn within_tolerance_difference_passes() {
    let path = std::env::temp_dir().join("m2s2_golden_tolerance.png");
    let golden = vec![10u8, 20, 30, 255];
    write_png(&path, &golden, 1, 1);
    let actual = vec![11u8, 19, 31, 255]; // every channel within 2
    assert!(assert_matches_golden(&actual, 1, 1, &path, 2).is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn missing_golden_without_update_flag_fails_clearly() {
    let path = std::env::temp_dir().join("m2s2_golden_missing_no_update.png");
    let _ = std::fs::remove_file(&path);
    std::env::remove_var("M2S2_UPDATE_GOLDEN");
    let err = assert_matches_golden(&[0u8, 0, 0, 255], 1, 1, &path, 2).unwrap_err();
    assert!(err.contains("M2S2_UPDATE_GOLDEN"), "error should mention the update flag: {err}");
}

#[test]
fn missing_golden_with_update_flag_writes_one() {
    let path = std::env::temp_dir().join("m2s2_golden_missing_with_update.png");
    let _ = std::fs::remove_file(&path);
    std::env::set_var("M2S2_UPDATE_GOLDEN", "1");
    let result = assert_matches_golden(&[1u8, 2, 3, 255], 1, 1, &path, 2);
    std::env::remove_var("M2S2_UPDATE_GOLDEN");
    assert!(result.is_ok());
    assert!(path.exists());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dimension_mismatch_fails_clearly_not_panics() {
    let path = std::env::temp_dir().join("m2s2_golden_dimension_mismatch.png");
    write_png(&path, &[0u8, 0, 0, 255, 0, 0, 0, 255], 2, 1);
    let err = assert_matches_golden(&[0u8, 0, 0, 255], 1, 1, &path, 2).unwrap_err();
    assert!(err.to_lowercase().contains("dimension") || err.contains("2") , "error should mention the dimension mismatch: {err}");
    let _ = std::fs::remove_file(&path);
}
```

(Tests that touch the shared `M2S2_UPDATE_GOLDEN` env var must not run concurrently with each other in a way that races — Rust runs tests in threads by default; either accept `cargo test`'s default behavior since each test uses a distinct temp file and only reads/writes the var around its own narrow window, or mark the two env-var tests `#[test]` as-is and note this if flakiness appears. Do not add `serial_test` or similar — out of scope for this plan.)

- [ ] **Step 3: Run to verify they fail**

Run: `cargo test --lib renderer::golden`
Expected: FAIL — `assert_matches_golden` doesn't exist.

- [ ] **Step 4: Implement `assert_matches_golden`**

In `src/renderer/golden.rs`, above the test module: if `std::env::var("M2S2_UPDATE_GOLDEN").is_ok()`, encode `actual` to `path` via the `png` crate (same encoder shape as the test helper) and return `Ok(())` unconditionally. Otherwise, open and decode `path` with `png::Decoder`/`Reader` — if the file doesn't exist or fails to decode, return `Err(format!("no golden image at {path:?} (or it failed to decode): run with M2S2_UPDATE_GOLDEN=1 to create it"))`. If the decoded image's `(width, height)` don't match the arguments, return `Err(format!("golden image dimension mismatch: golden is {}x{}, actual is {width}x{height}", decoded_width, decoded_height))`. Otherwise compare byte-for-byte with the given `tolerance`: iterate pixels, and for the first one where any channel's absolute difference exceeds `tolerance`, return `Err(format!("pixel ({x}, {y}) differs: golden {golden_px:?} vs actual {actual_px:?} (tolerance {tolerance})"))`. Return `Ok(())` if every pixel is within tolerance.

- [ ] **Step 5: Run to verify they pass**

Run: `cargo test --lib renderer::golden`
Expected: PASS.

- [ ] **Step 6: Run the full CPU gate suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/golden.rs src/renderer/mod.rs Cargo.toml Cargo.lock
git commit -m "feat(renderer): golden-image PNG comparison with per-channel tolerance"
```

---

### Task 5: End-to-end self-test, documentation, and spec outcome

Wires Tasks 1–4 together into the harness's own proof that the whole pipeline works, and documents how to use it. This task's final GPU step **requires the user** — generating the first golden image needs a real Vulkan driver this plan's executor does not have.

**Files:**
- Modify: `src/renderer/headless.rs` (the end-to-end self-test)
- Modify: `CLAUDE.md` (a short section: how to run the `#[ignore]`d GPU tests locally, how to regenerate golden images)
- Modify: `specs/019-headless-rendering-and-golden-image-tests.md` (Outcome), `ROADMAP.md` (if it lists this as engine-level infra, check it off or note status)

- [ ] **Step 1: Write the end-to-end self-test**

In `src/renderer/headless.rs`'s test module:

```rust
#[test]
#[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/headless_self_test.png"]
fn self_test_scene_matches_golden_with_zero_validation_errors() {
    let (vertices, indices) = lit_untextured_cube();
    let mut renderer = HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
    let camera = /* same fixed camera as the determinism test */;
    let objects = [Object { model: Mat4::IDENTITY, texture: None, specular: SpecularMaterial::default() }];
    let pixels = renderer.render_frame(&objects, camera).expect("render");

    let golden_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/golden/headless_self_test.png"));
    assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
    assert_eq!(/* instance.validation_error_count() via whatever accessor Task 3 exposed */, 0);
}
```

- [ ] **Step 2: Build (cannot run — no GPU here)**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test --all-targets`
Expected: all clean; this test is listed as ignored, not executed.

- [ ] **Step 3: Create the golden-image directory placeholder**

Run: `mkdir -p testdata/golden` and add a short `testdata/golden/.gitkeep` or `README.md` noting "generated golden images live here; see CLAUDE.md for how to regenerate them" — so the directory exists in git even before the first golden PNG is committed.

- [ ] **Step 4: Document the workflow in `CLAUDE.md`**

Add a short section (near the existing Quality gates section) explaining: GPU-dependent tests are `#[ignore]`d by default; run them locally with `cargo test -- --ignored` on a machine with a Vulkan driver; regenerate a golden image by running the specific test with `M2S2_UPDATE_GOLDEN=1 cargo test -- --ignored <test_name>`, then commit the resulting PNG under `testdata/golden/`.

- [ ] **Step 5: Commit the code/docs (not the golden image yet — that needs the user)**

```bash
git add src/renderer/headless.rs CLAUDE.md testdata/golden
git commit -m "test(renderer): end-to-end headless self-test; document the GPU-test workflow"
```

- [ ] **Step 6: Hand off to the user — generate the first golden image**

This plan's executor has no GPU and cannot complete this step. Report clearly that the user must run, on a machine with a Vulkan driver:
```bash
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless -- --ignored self_test_scene_matches_golden_with_zero_validation_errors
```
to generate `testdata/golden/headless_self_test.png`, then run the same command **without** `M2S2_UPDATE_GOLDEN` to confirm the self-test now passes, and finally run the other three `#[ignore]`d tests from Tasks 1 and 3 (`headless_instance_and_device_construct_and_tear_down`, `headless_renderer_constructs_and_tears_down`, `render_frame_is_deterministic_across_repeated_calls`) to confirm the whole harness works end to end. Only after the user confirms this should the golden PNG be committed and the spec's Outcome section + `ROADMAP.md` be updated to reflect verified (not just implemented) status.
