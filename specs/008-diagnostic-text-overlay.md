# 008: Diagnostic text overlay

**Status**: Draft proposal; not implemented.
**Roadmap item**: Phase 2, diagnostic text overlay.

## Goal

Display camera position, FPS, CPU frame time, and useful Vulkan state directly over the scene.
The overlay should make a bad camera, mismatched viewport, missing draws, or unexpected GPU selection
visible without switching to a terminal.

## Placement and scope

Place this in Phase 2 after 007. Implement after the first texture/sampler upload path so a font atlas
can reuse that infrastructure. If diagnostics are needed sooner, pull forward only the atlas-sized
image upload/sampler work and share it with later textures; do not create a second image uploader.

Initial metrics:
- Camera world position, supplied explicitly by the caller (the renderer does not own the camera).
- FPS and CPU frame interval in milliseconds, computed from caller-provided elapsed frame time.
- Selected physical-device name; swapchain extent, color format, and image count.
- Draw calls and submitted index count for the most recent submitted scene frame.
- Whether rendering is paused for zero extent or a swapchain recreation is pending.

Use a rolling timing window so the text remains readable; label CPU frame interval honestly. FPS is
application frame cadence, not a measurement of monitor presentation rate. Do not report skipped frames
as submitted GPU work. Initialize timing as unavailable until a valid positive sample is received.

## Non-goals

A general GUI, editable controls, graphs, font shaping/localization, a scene inspector, and device-loss
recovery. GPU timestamps, allocator memory budgets, pipeline statistics, and validation-message history
can follow later. GPU frame time must come from timestamp queries, never from CPU wall-clock timing.

## Proposed public API

The exact API should be finalized before implementation. Preserve existing render methods and keep
window/input ownership with the caller. Proposed additions:

```rust
pub enum DiagnosticOverlay { Hidden, Visible }

pub struct DiagnosticInput {
    pub camera_position: Vec3,
    pub frame_interval: std::time::Duration,
}

impl VulkanRenderer {
    pub fn set_diagnostic_overlay(&mut self, visibility: DiagnosticOverlay);
    pub fn update_diagnostics(&mut self, input: DiagnosticInput);
}
```

Default to hidden. When enabled, existing render methods append text after scene geometry; a clear-only
frame can still show diagnostics. The caller may map a key to visibility. Do not invert the view matrix
to guess camera state. Cache static device details; do not query device properties on every frame.

## Vulkan objects and lifetime

A minimal bitmap font atlas, image/view/sampler, descriptor resources, alpha-blended text pipeline,
and per-frame mapped text vertex/index buffers. Record overlay draws after scene draws, with depth
writes/testing disabled and viewport/scissor set explicitly. Size text in physical pixels with an explicit
scale supplied or selected by the caller for high-DPI readability.

Reuse the current color attachment inside the active dynamic-rendering scope if pipeline attachment
formats are compatible; otherwise specify a second load-preserving scope and its dependencies explicitly.
Bind a separate overlay pipeline/layout so camera/model state cannot accidentally affect text. Put font
upload and layout transitions on the existing shared resource path. Rewrite a frame slot's mapped text
buffers only after its fence signals. No per-frame wait-idle and no unbounded text allocation.

Destroy overlay buffers/images before their allocator and device, and apply constructor rollback.
Recreation must update extent-dependent state and rebuild pipelines if attachment formats change.

## Test plan

- Timing: valid samples, initial state, zero duration, rolling-window boundaries and reset.
- Formatting: known camera coordinates, stable metric labels/units, unavailable state.
- Counters: zero/one/multiple scene draws; exclude overlay draws from scene counts.
- Visibility: hidden emits no overlay commands; visible emits text commands after scene draws.
- Layout: bitmap glyph indices, clipping, bounded buffer capacity, scale and resize behavior.
- GPU: readable text over moving geometry, independent of scene depth, correct resize/minimize/restore,
  empty-scene overlay, no synchronization/validation errors, clean teardown.
- Run all repository quality gates; GPU-dependent checks stay opt-in outside ordinary CPU CI.

## Open questions

Before implementation, choose a small redistributable bitmap font and record its license/provenance;
finalize scaling and fixed buffer capacity; decide whether the first version includes GPU timestamps.
The recommended first version uses CPU timing only and labels it explicitly.

## Outcome

Pending design finalization and implementation. This captures the requested idea without changing 007.
