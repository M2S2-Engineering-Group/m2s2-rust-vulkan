# 009: Vulkan 1.3 core dynamic rendering

**Status**: Implemented and verified.
**Roadmap item**: not a Phase feature — a foundations simplification of spec 001's dynamic-rendering
choice (see [ROADMAP.md](../ROADMAP.md) Phase 1 "Render target strategy").
**Prerequisite**: specs 001–008 implemented; dynamic rendering already in use via the extension path.

## Goal

Require Vulkan **1.3** so dynamic rendering is a core feature, and drop everything the engine only
carries because it targets a 1.0 instance: the `VK_KHR_dynamic_rendering` device extension, its
pre-1.2 dependency chain, the `VK_KHR_get_physical_device_properties2` instance extension, the
properties2 feature-probe loader, and the `ash::khr::dynamic_rendering::Device` command loader.

No rendered output changes. This is purely how dynamic rendering is reached.

## Non-goals

No fallback path for pre-1.3 devices — requiring 1.3 with no alternative is the point. No change to
what gets drawn, to the pipeline, or to any public `VulkanRenderer` API. No bump beyond 1.3 (1.4+
devices satisfy a `>= 1.3` gate and need nothing extra).

## Rationale and risk

1.3 mandates dynamic-rendering support, so a `>= 1.3` device gate makes the capability guaranteed
and the extension/dependency bookkeeping redundant. The modern vulkan-tutorial.com path one consumer
follows already assumes 1.3 core.

Hard requirement, accepted consequence: on a genuinely 1.0-only loader, `create_instance` with
`apiVersion` 1.3 returns `ERROR_INCOMPATIBLE_DRIVER`; a device reporting `< 1.3` is skipped during
selection and, if none qualifies, initialization fails with the existing `InitializationError`.
Confirmed available on the development driver: both reported physical devices are `1.4.334` and
`1.3.350`.

## Changes

### `instance.rs`
- `ApplicationInfo::api_version`: `API_VERSION_1_0` → `API_VERSION_1_3`.
- Remove `ash::khr::get_physical_device_properties2::NAME` from the instance extension list; it was
  present only to run the 1.0-era feature probe (now gone). Surface, platform-surface, and
  debug-utils extensions are unchanged.

### `device.rs`
- `REQUIRED_DEVICE_EXTENSIONS` shrinks from six entries to one: `ash::khr::swapchain::NAME`. The
  dynamic-rendering extension and its `depth_stencil_resolve → create_renderpass2 → {multiview,
  maintenance2}` chain are all core in 1.3. Update the constant's doc comment accordingly.
- In `pick_physical_device`, replace the `get_physical_device_properties2` feature probe with a
  device API-version gate: skip any device whose `properties.api_version` is `< 1.3`. Extract the
  comparison into a pure helper `meets_min_api_version(device_version: u32) -> bool` (mirrors
  `score_device_type`'s testable-pure-logic shape) so it is unit-testable without a driver.
- Keep enabling dynamic rendering as a feature: the
  `PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true)` push_next on
  `DeviceCreateInfo` stays — 1.3 guarantees support but the feature must still be turned on.
- Remove the `dynamic_rendering_supported` helper and its test; add a `meets_min_api_version` test.

### `command.rs`
- Delete the `dynamic_rendering: ash::khr::dynamic_rendering::Device` field from `FrameContext`.
- `FrameContext::new` drops its `instance: &VulkanInstance` parameter (it existed only to build that
  loader) and the associated comment.
- `cmd_begin_rendering` / `cmd_end_rendering` are called directly on `ash::Device` (`dev`) instead of
  through the loader.

### `mod.rs`
- Update the single `FrameContext::new(&instance, &device, …)` call site to drop `&instance`, and the
  adjacent comment that explained the extra argument.

## Test plan

CPU-only (CI has no driver; nothing here calls `create_instance`):
- `meets_min_api_version` accepts `1.3.x` and `1.4.x` encodings and rejects `1.2.x`, `1.1.x`, `1.0.x`
  (use `vk::make_api_version` to build the inputs).
- Existing suite (currently 48 tests) still passes with the `dynamic_rendering_supported` test
  swapped for the version-gate test.

Manual GPU regression (before/after must be identical):
- `cargo run --example cube` with synchronization validation enabled renders the two cubes +
  overlay with zero validation errors, visually unchanged from the pre-change build.

All four repository quality gates pass clean.

## Open questions

None. If a future consumer must support a pre-1.3 target, that reintroduces a fallback and is its own
spec; this one deliberately does not provide one.

## Outcome

Implemented as specified. The instance now requests `API_VERSION_1_3`; `REQUIRED_DEVICE_EXTENSIONS`
is just `swapchain`; device selection gates on `meets_min_api_version` (≥ 1.3); the
`get_physical_device_properties2` instance extension, the properties2 feature-probe loader, and the
`ash::khr::dynamic_rendering::Device` command loader are all gone. `FrameContext::new` dropped its
`&VulkanInstance` parameter, and `cmd_begin_rendering`/`cmd_end_rendering` are core `ash::Device`
calls. Dynamic rendering is still enabled via `PhysicalDeviceDynamicRenderingFeatures`.

All four quality gates pass; the suite is 48 tests (the `dynamic_rendering_supported` test was
replaced by `requires_vulkan_1_3_or_newer`). GPU regression: `cargo run --example cube` with
synchronization validation enabled rendered the two cubes and overlay correctly with clean output —
visually unchanged from the extension-path build, confirming the core path is equivalent.
