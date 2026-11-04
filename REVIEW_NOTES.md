# Rendering correctness review

This records the review fixes made before spec 007 and their verification follow-up.

## Findings and resolutions

1. **Shared depth attachment across frames.** Clearing depth and using `UNDEFINED` discards contents,
   but does not order overlapping GPU accesses. The barrier now orders prior depth writes at both
   early and late fragment tests before subsequent depth reads/writes at those stages. This retains
   one depth image and serializes its use; separate attachments can be considered if profiling warrants it.
2. **Depth replacement on resize.** Create the new attachment before destroying the old one, so an
   allocation failure preserves valid owned handles and shutdown cannot destroy stale handles.
3. **Different graphics/present queue families.** Select concurrent swapchain sharing with both family
   indices when they differ; keep exclusive sharing when they match.
4. **Partial initialization leaks.** Add scoped handle rollback and collection rollback. Cover failed
   buffer/image allocation or binding, staging upload, transient copy commands/fences, shader/pipeline
   creation, uniform/descriptors, frame synchronization, image views/swapchain/surfaces, and debug
   messenger creation. Renderer initialization explicitly frees completed allocator-backed resources
   before the allocator/device unwind. Present semaphore replacement also creates before destroying.
5. **Initial depth extent.** Use the selected swapchain extent, matching the resize path, rather than
   the caller's potentially clamped window dimensions.
6. **Dynamic-rendering feature support.** Query the feature through the Vulkan 1.0 KHR properties2
   entry point during suitability checks; skip devices reporting false before device creation.
7. **Additional runtime finding: acquire/layout transition dependency.** Synchronization validation
   reported a WRITE_AFTER_READ hazard from acquire to the color layout transition. The transition's
   source stage now matches the acquire semaphore's `COLOR_ATTACHMENT_OUTPUT` wait stage. One named
   helper supplies both scopes so they cannot drift.

References: Khronos [synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
and [dynamic-rendering feature query](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceDynamicRenderingFeatures.html).

## Verification

- CPU regression suite: 31 tests. Added rollback success/failure and collection partial-failure checks,
  shared-depth stage coverage, acquire-stage coverage, queue-family sharing, and feature support checks.
- Formatting, Clippy with warnings denied, all-target build/test, and diff whitespace checks pass.
- Ran `env VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo run --example cube`
  outside the sandbox. The first run exposed finding 7; the rerun rendered continuously without emitted
  validation errors. The local loader exposes Apple M5 Vulkan implementations.
- The sandbox itself cannot access Metal; `vulkaninfo --summary` succeeded outside it.

Limits: CPU tests exercise cleanup control flow rather than injecting failures into a real Vulkan driver.
The runtime check does not establish correctness on hardware with separate queue families. Repeated
interactive resize/minimize/restore and graceful window-close teardown still need manual verification;
the smoke run was stopped through the terminal. Recovery from device loss or a retired swapchain after
failed recreation is outside this fix: callers should treat Vulkan errors as fatal and tear down safely.
Spec 008 subsequently added pipeline recreation when swapchain attachment formats change.

## Next

[Spec 007](specs/007-multiple-objects-and-push-constants.md) now implements independently transformed
copies of one mesh. Its verification used a temporary automated cube example to exercise two window
resizes, minimize/restore, clear-only and single-object frames, and normal event-loop shutdown. That run
exited successfully without emitted synchronization-validation errors. This supplements the original
review smoke run; rendered-pixel correctness and separate-family hardware remain unverified.
[Spec 008](specs/008-diagnostic-text-overlay.md) implements reusable text drawing and the Phase 2 diagnostic overlay.

## Spec 008 implementation follow-up

- Added per-frame `TextDraw` submissions with physical-pixel placement, font scale, linear RGBA,
  newline support, viewport clipping, and an explicit 2048-visible-glyph limit. Existing rendering APIs
  forward empty text; text resources are created lazily.
- Kept CPU `Diagnostics` separate from the text renderer. It formats explicit camera position, a
  rolling 60-sample FPS/CPU interval, cached GPU name, swapchain details, and last-submitted scene counts.
  Scene counts exclude overlay draws and skipped acquisition. GPU timing remains deferred.
- Shared pipeline construction and one-time command submission between geometry and text. Font upload
  uses a reusable sampled-R8 image path; bundled public-domain font provenance and regeneration are
  documented in `assets/README.md`.
- Added per-frame fence-protected text buffers, alpha-blended drawing after geometry with depth
  disabled, constructor rollback, ordered teardown, and format-dependent scene/text pipeline recreation.
- Updated the cube example to show diagnostics by default with F3 visibility control.

Verification: all 48 CPU tests, formatting, Clippy with warnings denied, all-target build, shader
validation, and whitespace checks pass. A GPU probe exercised resize, hide/show, minimize/restore,
clear-only and single-object text, submitted-count assertions, and normal shutdown without emitted
synchronization-validation errors. A captured Vulkan frame was visually inspected for readable text.
The temporary probe was removed. Cross-platform runtime checks and injected GPU-allocation failures
remain unverified; the full API/lifetime contract and outcome are recorded in spec 008.
