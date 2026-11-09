# 019: Headless offscreen rendering and golden-image tests

**Status**: Implemented and verified (CPU gates pass; GPU acceptance confirmed by user).
**Roadmap item**: [Engine-level concerns](../ROADMAP.md#phase-5-engine-level-concerns) — test
infrastructure, pulled forward as a prerequisite for spec 018's GPU acceptance tests.
**Dependencies**: Spec 009 (Vulkan 1.3, dynamic rendering core), spec 012 (resource-ownership
patterns this reuses: `Option` teardown, consuming `destroy`, the `Cleanup`/`collect_resources`
rollback helpers). Independent of specs 013–018's shading work — this spec only builds the harness
that renders *whatever scene a test gives it* and compares pixels; it contains no lighting math of
its own. Spec 018 (point lighting) is this harness's first consumer.

## Goal

Give the crate a way to render a scene with no window, no swapchain, and no display server at all,
read the result back to the CPU, and compare it against a saved reference image ("golden image")
within a small per-channel tolerance — so any future rendering feature can be verified by an
automated test instead of a human eyeballing an example. This is general-purpose test
infrastructure, not a one-off for point lighting.

## Non-goals

- **No CI wiring yet.** The CI runner has no GPU today. These tests must not run under
  `cargo test --all-targets` (the command the pre-commit hook and CI both already run) — they are
  `#[ignore]`d by default and run locally via `cargo test -- --ignored`. Wiring them into CI is
  deferred until a software Vulkan implementation (`lavapipe`) is available there; this spec's job
  is to make that a config change later, not a redesign. See "Why true headless" below for why that
  constrains the design now.
- **No new public API.** Headless construction and the readback/comparison helpers are
  `pub(crate)`-only. No consumer outside this crate gets a new way to render to a texture; this is
  test tooling, not a product feature. (Resolved decision — see Open questions.)
- **No perceptual/structural image diffing** (SSIM, perceptual hashing, etc.). A simple per-pixel,
  per-channel absolute-difference tolerance is the whole comparison model. If that proves
  insufficient once real cross-driver testing happens (local vs. `lavapipe`), a follow-up spec
  revisits it — not speculative generality now.
- **No retroactive golden images for specs 011–013.** Adding golden-image coverage for existing
  directional/specular lighting is optional future work a maintainer can do once this harness
  exists; not required by this spec.
- **No multi-GPU/multi-driver test matrix, no automatic golden-image regeneration on CI, no
  diff-visualization UI** beyond writing a failure artifact next to the golden file for a human to
  open.

## Why true headless (not a hidden real window)

The cheapest way to get *something* working would be to keep using the existing `with_window` +
real swapchain path, point a real (possibly off-screen-positioned) OS window at it, and copy the
presented image back instead of actually showing it. That was considered and rejected: a real OS
window needs a real display server (X11/Wayland/etc.) to create, and a headless CI container with
`lavapipe` installed typically has **no display server at all** — `lavapipe` supplies the Vulkan
*implementation*, not a windowing system. Building on a real window now would work locally but hit
a wall the day CI wiring is attempted, defeating this spec's stated purpose. The harness must not
create a window, a surface, or depend on any windowing-system extension actually being backed by a
live display server.

This constraint reaches the **instance**, not just the device. `VulkanInstance::new()` (the existing
constructor every other render path uses) unconditionally requests `VK_KHR_surface` plus a
platform surface extension (`xlib_surface`/`wayland_surface` on Linux, `metal_surface` on macOS) —
`vkCreateInstance` fails outright if the loader doesn't advertise one. On a real dev machine this
is harmless (the loader always advertises its platform's surface extensions), but a minimal
headless Linux CI container may lack the client-side WSI libraries entirely, in which case the
loader never advertises `xlib_surface`/`wayland_surface` and instance creation fails before any
headless device logic runs — silently defeating the "config change later, not a redesign" goal.
`HeadlessRenderer` therefore needs its own `VulkanInstance::new_headless()` that requests only
`VK_EXT_debug_utils` (debug builds) and nothing surface-related; it never calls `create_surface`,
so it needs none of `VK_KHR_surface`'s functionality either.

## Assertable validation errors (resolved during review)

A review of the first full draft pointed out that "zero validation errors" and "no allocator leak
reports" were stated as acceptance criteria but weren't actually things a test could fail on — they
were log lines a human would have to read, same as every prior spec's manual GPU check. Since this
spec's whole point is replacing eyeballing with automation, that gap is fixed for the validation
side: the existing debug-messenger callback (`instance.rs`'s `debug_callback`, currently
log-only) gains a null-checked `pUserData` path — when non-null, it's treated as `*const
AtomicU32` and incremented on `ERROR` severity, in addition to its existing logging. The normal
`with_window`/`unsafe new` path passes a null user-data pointer (unchanged behavior, nothing reads
it). `new_headless` instead owns an `Arc<AtomicU32>`, passes a pointer to it as `pUserData`, and
exposes the current count via `validation_error_count()`. `HeadlessRenderer`'s GPU tests assert
this is `0` after each render, alongside the pixel comparison — a genuine automated check, not a
documented aspiration. Allocator-leak detection stays a documented limitation for now (`Allocator`
has no `Drop` impl today and gpu-allocator's own shutdown diagnostics are log-only); worth a
follow-up if it becomes a real need, not scope for this spec.

## Public API

Nothing public. Every new type/function below is `pub(crate)` and exists **only in test builds**:
the whole new module tree is declared `#[cfg(test)]` at the module-declaration site in `mod.rs`
(`#[cfg(test)] pub(crate) mod headless;`), so none of it is compiled into a normal `cargo build` —
zero dead-code risk, zero change to the production binary, and no risk to the real `VulkanRenderer`.

**`HeadlessRenderer` is a distinct type, not a mode of `VulkanRenderer`.** `VulkanRenderer` has a
non-optional `swapchain: ManuallyDrop<VulkanSwapchain>` field, and `render_objects_with_text`,
`resize`, `stats`, and `Drop` all reach into it directly — retrofitting an "or there's no swapchain"
branch into that struct would mean touching the reviewed, GPU-verified production path for the sake
of test infrastructure, exactly what this spec's "why true headless" reasoning already argues
against doing carelessly. Instead, `HeadlessRenderer` is its own small struct, parallel to
`VulkanRenderer` but independent of it, owning only what a one-shot headless render needs: device,
allocator, pipeline, depth, the camera/light/texture uniform infra, and an `OffscreenTarget` in
place of a swapchain. It does not implement `resize`, `stats`, texture creation after construction,
or anything else `VulkanRenderer` has that a single fixed-scene test render doesn't need.

```rust
// src/renderer/instance.rs
impl VulkanInstance {
    /// Headless construction: requests only `VK_EXT_debug_utils` (debug builds) — no
    /// `VK_KHR_surface`, no platform surface extension. Never calls `create_surface`. Its debug
    /// messenger (debug builds only) counts ERROR-severity messages in an atomic counter instead
    /// of only logging them, so a test can assert zero validation errors rather than relying on a
    /// human reading the log — see "Assertable validation errors" below.
    pub(crate) fn new_headless() -> Result<Self>;

    /// Validation ERROR messages observed since construction (debug builds; always 0 in release,
    /// where validation layers aren't loaded). `HeadlessRenderer`'s GPU tests assert this is 0
    /// after rendering, alongside the pixel comparison.
    pub(crate) fn validation_error_count(&self) -> u32;
}

// src/renderer/device.rs
impl VulkanDevice {
    /// Headless construction: picks a device by graphics-queue-family and API version only — no
    /// surface, no present-queue requirement, no VK_KHR_swapchain extension. `present_queue` and
    /// `present_queue_family` are set equal to the graphics queue/family (there is nothing to
    /// present to); existing swapchain-based code never runs for a device built this way. Takes a
    /// `VulkanInstance` built via `new_headless` (above) — never a surface-carrying one.
    pub(crate) fn new_headless(instance: &VulkanInstance) -> Result<Self>;
}

// src/renderer/headless.rs (new file, #[cfg(test)] module)
/// A plain color VkImage + view (no swapchain, no surface) sized like a swapchain image, plus the
/// host-visible buffer its contents get copied into after rendering. **`R8G8B8A8_UNORM`**, not the
/// swapchain's `B8G8R8A8_SRGB` — see "Format choice" below for why. Tracks its own current image
/// layout *and* the access/stage scope that last wrote it, so `render_frame` can be called more
/// than once on the same instance with a correctly-scoped barrier each time (not just the right
/// `old_layout` — see "Repeated-call barriers" below).
pub(crate) struct OffscreenTarget { /* image, view, allocation, readback buffer, extent, current_layout, last_access */ }

impl OffscreenTarget {
    pub(crate) fn new(device: &VulkanDevice, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self>;
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator);
}

/// A minimal, self-contained renderer for offscreen test rendering. Owns device/allocator/pipeline/
/// depth/uniform infra independent of `VulkanRenderer` — see the note above on why this is a
/// separate type. Constructed once per test (or reused across several `render_frame` calls within
/// one test, e.g. the determinism check); not meant to outlive the test that builds it.
pub(crate) struct HeadlessRenderer { /* device, allocator, pipeline, depth, uniforms, texture/light infra, OffscreenTarget */ }

impl HeadlessRenderer {
    pub(crate) fn new(extent: (u32, u32), vertices: &[Vertex], indices: &[Index]) -> Result<Self>;

    /// Render one frame into the offscreen target (no acquire, no present, no frames-in-flight — a
    /// single synchronous submit-and-wait) and return the captured pixels as tightly-packed RGBA8
    /// rows, top-to-bottom, matching the format a PNG encoder expects. May be called more than once
    /// on the same instance (see `OffscreenTarget`'s layout tracking).
    pub(crate) fn render_frame(&mut self, objects: &[Object], camera: uniform::Camera) -> Result<Vec<u8>>;
}

// src/renderer/golden.rs (new file, test-only: #[cfg(test)])
/// Compare `actual` (RGBA8, `width`x`height`) against the PNG at `path`. If the
/// `M2S2_UPDATE_GOLDEN` environment variable is set, writes `actual` to `path` as the new
/// reference and returns Ok(()) unconditionally (regenerate mode). Otherwise decodes the existing
/// PNG and fails if any pixel's any channel differs from `actual` by more than `tolerance`
/// (absolute, 0-255 scale), reporting the first differing pixel's coordinates and values.
pub(crate) fn assert_matches_golden(
    actual: &[u8], width: u32, height: u32, path: &std::path::Path, tolerance: u8,
) -> Result<(), String>;
```

## Vulkan objects and lifetime

- **`VulkanDevice::new_headless`**: enumerates physical devices exactly as `pick_physical_device`
  does today, but the per-device checks drop `has_adequate_swapchain_support` and the
  present-queue half of `find_queue_families` entirely (a new `find_graphics_queue_family` — a
  graphics-capable queue family is the only requirement). `supports_required_extensions` is not
  called; `device_extensions` passed to `create_device` is empty (no `VK_KHR_swapchain` — nothing
  uses it). Dynamic rendering is still enabled via `push_next`, exactly as today, since the draw
  path needs it regardless of presentation.
- **`OffscreenTarget`**: one `vk::Image` (`COLOR_ATTACHMENT | TRANSFER_SRC`, OPTIMAL tiling, same
  sample-count conventions as the swapchain images) + its view, built the same way `depth.rs`'s
  `DepthResources` builds its own image (same allocator, same `Cleanup`/rollback pattern from spec
  012, and — unlike the first draft of this spec — the **same format-support query pattern**
  `depth.rs` already uses: probe `get_physical_device_format_properties` for
  `COLOR_ATTACHMENT_OPTIMAL` support rather than asserting a format with no query, since this
  image has no surface to source a guaranteed-supported format from) — plus one host-visible
  `Buffer` (reusing `buffer::Buffer`'s existing `new_host_visible_mapped`) sized
  `width * height * 4` bytes to receive the copy-back. Starts in `vk::ImageLayout::UNDEFINED` and
  tracks both its current layout and the access/stage scope of whatever last wrote it, so
  `HeadlessRenderer::render_frame` is safe to call repeatedly (see "Repeated-call barriers" below).

  **Format choice**: `R8G8B8A8_UNORM`, not the swapchain's `B8G8R8A8_SRGB`. The first draft of this
  spec picked the sRGB format "to match what a window would show," but that actively works against
  this harness's actual goal: an sRGB-encoded store applies a nonlinear curve whose slope is
  steepest near black, which *amplifies* small cross-driver floating-point differences exactly in
  dark regions (e.g. a clear color) — the opposite of what a tolerance-based, cross-driver-portable
  comparison wants. This harness exists to verify rendering *correctness*, not to produce a
  display-accurate preview; a linear UNORM target stores the computed color directly, with no
  encode-curve amplification, and is also more broadly guaranteed as a renderable format (the
  swapchain's SRGB choice came from a surface format query this image doesn't have access to).

- **Depth**: reuse `depth::DepthResources` unchanged — it is already constructed from an explicit
  extent with no swapchain dependency.
- **`emit_object_draws`/`ObjectCommand` become `pub(super)`** (currently module-private to
  `command.rs`, used only by `record` and `command.rs`'s own tests). This is a real, if small, edit
  to `command.rs` — the first draft of this spec undersold it as "command.rs untouched"; the
  visibility widening is required for `headless.rs` to call the same draw-emission logic rather
  than duplicating it.
- **One-shot frame recording**: a new, separate function on `HeadlessRenderer` (not a modification
  of `FrameContext::record`, which is swapchain-acquire/present-shaped) that: transitions the
  offscreen color image to `COLOR_ATTACHMENT_OPTIMAL`, begins dynamic rendering against it (and the
  depth attachment), binds the pipeline and the camera/light/texture descriptor sets exactly as
  `record` does today, calls the **same** shared `emit_object_draws` (now `pub(super)`; spec
  007/010/013's existing, already-tested draw-emission function — the one real risk surface,
  correctness of what gets drawn, is reused verbatim, not reimplemented), ends rendering,
  transitions the color image to `TRANSFER_SRC_OPTIMAL`, and issues one `vkCmdCopyImageToBuffer`
  into the readback buffer. Submitted once via a pattern mirroring `upload::submit_once` (create a
  transient command pool/buffer, submit, wait on a fence, done) — no frames-in-flight, no
  semaphores: a single-shot render has nothing to overlap with.

  **Repeated-call barriers**: tracking only `old_layout` (as the previous revision of this spec
  described) is not sufficient — a correct barrier also needs the right *source* access mask and
  pipeline stage, and those differ by call. The first call's color-image barrier goes
  `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL` with an empty source access mask from `TOP_OF_PIPE` (there
  is no prior access to wait on). Every later call's first barrier goes
  `TRANSFER_SRC_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL` with source access `TRANSFER_READ` from stage
  `TRANSFER` (the prior call's `vkCmdCopyImageToBuffer` is the actual last writer/reader).
  `OffscreenTarget` tracks which case applies and the one-shot recording function picks the correct
  barrier source scope accordingly — getting this wrong doesn't fail to render, it passes a
  `SYNC-HAZARD` validation error, which is exactly the acceptance criterion this harness is meant
  to check.
- **Readback byte order**: `R8G8B8A8_UNORM`'s in-memory channel order is already R,G,B,A — unlike
  the swapchain's `B8G8R8A8_SRGB`, no swizzle is needed before handing the bytes to the PNG encoder.
  This is a side benefit of the format change above, not a reason for it.
- **Teardown**: `OffscreenTarget::destroy` and the headless renderer's `Drop` follow spec 012's
  established pattern exactly (consuming `destroy(self, ..)`, `Option` fields moved out via
  `take()`) — no new teardown design, reuse of the existing one.

## Interactions

- `instance.rs`: `VulkanInstance::new_headless` (no surface extensions) and
  `validation_error_count`. Gate both `#[cfg(test)]` — no caller outside the new module tree. The
  existing `debug_callback` gains the null-checked atomic-increment path described above; this one
  function IS shared with the production debug messenger, but the change is additive and a no-op
  when `pUserData` is null (the production path's behavior is unchanged).
- `device.rs`: `new_headless` + `find_graphics_queue_family`, sharing `score_device_type`/
  `meets_min_api_version` with the existing path. Gate **both** new items `#[cfg(test)]` — they
  have no caller outside the `#[cfg(test)]` module tree, so both would otherwise trip a dead-code
  warning in normal builds, not just the one the first draft of this spec named.
- `command.rs`: widen `emit_object_draws` and `ObjectCommand` from private to `pub(super)` (see
  "Vulkan objects" above — this is a real, small edit, not "untouched"). No behavior change to
  either; `FrameContext`/`record` are otherwise untouched.
- New `headless.rs` (whole module `#[cfg(test)]`): `OffscreenTarget`, `HeadlessRenderer` — a
  self-contained renderer independent of `VulkanRenderer` (see Public API above for why). Calls the
  now-`pub(super)` `emit_object_draws` via its own one-shot recording function.
- New `golden.rs` (test-only): `assert_matches_golden`, reachable from `#[cfg(test)]` modules
  anywhere in the crate.
- `Cargo.toml`: new **dev-dependency** `png` (lightweight pure encode/decode; not a runtime
  dependency — this is test tooling, consistent with spec 010's "no image-file decoding in this
  crate" non-goal, which was about the *library's* scope, not test infrastructure).
- New directory `tests/golden/` (or `src/renderer/testdata/golden/`, builder's call in the plan) for
  committed reference PNGs, plus any failure-artifact output path convention.
- No existing public API, shader, or production render-path code changes. The swapchain-based
  `with_window`/`unsafe new` paths are untouched.

## Test plan

CPU-only (always runs under `cargo test --all-targets`):
- `assert_matches_golden`'s pure comparison logic (no GPU needed to test the comparison function
  itself): identical images at tolerance 0 pass; a single out-of-tolerance pixel fails and reports
  its coordinates; a within-tolerance difference passes; a missing golden file with
  `M2S2_UPDATE_GOLDEN` set writes a new one and returns `Ok`; a missing golden file *without* the
  env var set fails with a clear "no golden image, run with M2S2_UPDATE_GOLDEN=1" message rather
  than panicking.
- `meets_min_api_version`/`score_device_type` reuse — no new tests needed, already covered.

GPU (local-only, `#[ignore]`d by default, run via `cargo test -- --ignored`, needs a real Vulkan
driver and is this spec's actual acceptance gate):
- `HeadlessRenderer::new` constructs successfully and tears down cleanly (no validation
  errors, no `gpu-allocator` leak reports) on this machine's driver. On the current dev machine
  (MoltenVK) specifically confirm device creation succeeds with the empty extension list this spec
  proposes — don't assume `VK_KHR_portability_subset` isn't required without checking
  (`VulkanDevice::new`'s existing swapchain-only extension list already implies it works, but
  verify rather than inherit the assumption silently).
- `HeadlessRenderer::render_frame` on a known scene — the existing lit cube geometry, **untextured**
  (`texture: None` on every object; texturing itself is already covered by spec 010's existing unit
  tests and isn't this harness's concern to re-verify, and `HeadlessRenderer` deliberately has no
  texture-creation API — see "Public API" above), with a fixed camera and fixed lights, no
  animation — produces a stable image: rendering it twice in a row produces byte-identical output
  (proves the one-shot path has no nondeterminism of its own, before any golden-file comparison
  enters the picture).
- The same known scene, compared against a golden PNG committed to the repo, passes within the
  chosen tolerance, **and** `validation_error_count()` is `0` after the render — this is the
  harness's own end-to-end self-test, proving the whole pipeline (render → readback → PNG decode →
  compare → assert-zero-validation-errors) before spec 018 depends on it.
- Running with `M2S2_UPDATE_GOLDEN=1` regenerates the golden file and the subsequent normal run
  passes against it.

Run all four repository quality gates (`fmt`, `clippy -D warnings`, `build --all-targets`,
`test --all-targets`) — the `#[ignore]`d GPU tests must not affect this, by construction. Document
in `CLAUDE.md` or a short note in this spec's Outcome how to run the GPU tests locally and how to
regenerate golden images.

## Open questions

Resolved during design (recorded here, not left open):
- **Public vs. internal API**: internal-only (`pub(crate)`). No new product-facing rendering mode.
- **Comparison model**: golden-image (saved reference PNG) with a small per-channel absolute
  tolerance, not per-pixel analytical expected values and not perceptual diffing.
- **Golden-image update workflow**: an environment variable (`M2S2_UPDATE_GOLDEN`) that, when set,
  overwrites the golden file with the current render instead of comparing against it.
- **CI**: not wired up now; deferred until `lavapipe` is available in CI. The true-headless design
  (no window, no display-server dependency, a headless-only `VulkanInstance`) removes the
  *architectural* blocker so wiring-up is a config/CI-script change, not a redesign — but the
  *golden images themselves* are driver-specific: a reference captured on this machine's driver
  will likely need regenerating against `lavapipe`'s output once that's set up, since different
  Vulkan implementations won't rasterize identically even within tolerance. That's a one-time
  `M2S2_UPDATE_GOLDEN=1` run against the new driver, not a design change, but it is a step, not a
  pure flip-the-gate config edit.

Left to the implementation plan: the exact tolerance value (propose starting at 2/255 per channel
and adjusting if the local driver proves noisier than that across repeated identical renders — the
"render twice, compare byte-for-byte" self-test above will reveal whether any tolerance is even
needed locally before cross-driver testing happens); the exact golden-file directory location and
naming convention.

## Outcome

Implemented across five tasks, entirely `#[cfg(test)] pub(crate)` — zero public API, zero
production-build footprint. `VulkanInstance::new_headless()` requests no surface/platform-surface
extensions and `VulkanDevice::new_headless()` requires only a graphics-capable queue family (no
present support, no `VK_KHR_swapchain`); both gained an assertable `validation_error_count()` via a
null-checked debug-messenger `pUserData` pointer to an `AtomicU32` (production `with_window`/`unsafe
new` pass a null pointer and are unaffected — confirmed byte-for-byte unchanged by the final review).
`OffscreenTarget` (`src/renderer/headless.rs`) selects `R8G8B8A8_UNORM` preferred / `B8G8R8A8_UNORM`
fallback via the same queried-not-asserted pattern as `depth.rs`, and tracks render state with a
single `bool` (`has_rendered`) — sufficient because only two barrier states exist: never-rendered
(`UNDEFINED → COLOR_ATTACHMENT_OPTIMAL`, empty access, `TOP_OF_PIPE`) and previously-rendered
(`TRANSFER_SRC_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL`, `TRANSFER_READ`, `TRANSFER`). `HeadlessRenderer`
mirrors `VulkanRenderer`'s exact construction order, field-wrapper split (`ManuallyDrop` for
device/instance/pipeline, `Option` for the rest), and reverse-order `Drop`; its one-shot command
submission mirrors `upload::submit_once`'s `Cleanup`-guard/`SubmitState` pattern line-for-line, so a
transient command pool is destroyed even if recording fails partway through. `golden.rs` implements
`assert_matches_golden` — per-channel absolute tolerance (default 2/255), dimension check before the
pixel loop, an `M2S2_UPDATE_GOLDEN` env var to regenerate, and one combined "missing or failed to
decode" error message naming that env var in both cases (a decode-only variant of this message
initially missed the env-var mention; caught by task review and fixed in the same branch). Golden
PNGs live under `testdata/golden/` at the repo root. The end-to-end self-test
(`self_test_scene_matches_golden_with_zero_validation_errors`) wires every piece together: headless
instance/device → `HeadlessRenderer` → `render_frame` → `assert_matches_golden` →
`validation_error_count() == 0`.

Verified (CPU): 73 unit tests pass, 5 `#[ignore]`d (GPU-dependent), plus 2 example-crate tests and 3
doctests; all four quality gates (`cargo fmt --all -- --check`, `cargo clippy --all-targets
--all-features -- -D warnings`, `cargo build --all-targets`, `cargo test --all-targets`) clean.

**GPU acceptance: confirmed by the user.** All 5 previously-`#[ignore]`d tests
(`headless_instance_and_device_construct_and_tear_down`,
`headless_renderer_constructs_and_tears_down`, `render_frame_is_deterministic_across_repeated_calls`,
`offscreen_target_constructs_tracks_render_state_and_tears_down`, and
`self_test_scene_matches_golden_with_zero_validation_errors`) pass on real Vulkan hardware with zero
validation errors. The first golden image, `testdata/golden/headless_self_test.png`, was generated
with `M2S2_UPDATE_GOLDEN=1` and is committed to the repo — regenerating it against a different driver
(e.g. once `lavapipe` is wired into CI, per this spec's deferred-to-later-section note above) is a
one-time `M2S2_UPDATE_GOLDEN=1` re-run, not a design change.

An independent whole-branch review (model: Opus, range of all 5 tasks) found 0 Critical, 0 Important
issues, and confirmed the production footprint is provably zero and every item in the plan's Review
Focus section holds against the diff. Three Minor findings were deferred as non-blocking: `golden.rs`
assumes a decoded golden is 8-bit RGBA with no `color_type`/`bit_depth` guard (a non-RGBA8 golden
would panic rather than error cleanly — low risk since goldens are only ever self-generated via
`M2S2_UPDATE_GOLDEN`); the readback buffer uses `MemoryLocation::CpuToGpu` rather than the more
idiomatic `GpuToCpu` for a GPU→CPU read (functionally correct — `CpuToGpu` is `HOST_COHERENT` under
this project's gpu-allocator 0.28 Vulkan backend, confirmed without GPU access by code inspection);
and the release-mode `validation_error_count() → 0` body is never actually compiled by the standard
gates (debug-only), so this spec's claim that a CPU test covers that path is slightly overstated.

Local-only for now, CI deferred until `lavapipe` is wired in, per this spec's resolved decisions.
