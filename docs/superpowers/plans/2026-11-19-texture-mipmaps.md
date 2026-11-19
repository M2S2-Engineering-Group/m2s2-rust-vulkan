# Texture Mipmaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let callers opt a texture into a full mip chain (`Mipmaps::Generate`), generated on the
GPU via linear blits at upload time, sampled during minification; `Mipmaps::Disabled` (the default)
keeps today's single-level behavior byte-for-byte.

**Architecture:** Extend the existing shared `SampledImage` (`image.rs`) with a named internal mip
policy computed once per `create_texture_with_options` call from queried device capabilities (pure,
unit-testable helper) and real per-texture dimensions. A policy of more than one level allocates all
levels up front, uploads the base level, then records `level_count - 1` blit+barrier steps in the
same one-time submission `image.rs` already uses. `create_texture` (existing, unchanged signature)
delegates to `create_texture_with_options` with `TextureOptions::default()` (`Mipmaps::Disabled`), so
every existing caller keeps its current behavior with zero code changes.

**Tech Stack:** Rust, `ash` 0.38 (Vulkan 1.3 core dynamic rendering), `gpu-allocator`, existing
`image.rs`/`upload.rs`/`texture.rs` modules.

**Spec:** [specs/014-texture-mipmaps.md](../../../specs/014-texture-mipmaps.md) (including its
"Resolved decisions (2026-11-19)" section — read both; the plan argues from the whole file).

## Global Constraints

- `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo build --all-targets`, and `cargo test --all-targets` must all pass clean after **every**
  commit — the pre-commit hook and CI both run the full crate, not just the changed files. A
  signature/shape change in one task must not break a second caller the task didn't anticipate; see
  the plan's Review Focus and the project's `feedback_sdd_plan_must_keep_crate_compiling` lesson.
- `TextureOptions::default()` is `Mipmaps::Disabled`; `create_texture`'s existing one-level behavior
  must not change in any observable way (same image/view/sampler shape, same bytes uploaded).
- `Generate` is an explicit opt-in. A device/extent combination that cannot support the multi-level
  blit path fails `create_texture_with_options` with a descriptive `VulkanError::ValidationError`
  *before* any GPU submission — never a silent one-level fallback, a different format, or a
  different device/filtering choice. No `TextureId`/descriptor-store entry is published on failure.
- No new device feature or crate dependency. No anisotropic filtering, compressed/array/cube/3D
  images, CPU/compute mip generation, authored mip chains, streaming, texture unloading, or a general
  sampler-configuration API.
- The R8 font atlas (`text.rs`) and the white default texture both keep their existing one-level,
  nearest-filtering behavior unchanged — they must not acquire the multi-level blit capability
  requirement even internally.
- GPU-dependent tests stay `#[ignore]`d (the project's CI runner has no Vulkan driver); ordinary
  `cargo test --all-targets` must not require one.

## Review Focus

- **Odd/non-power-of-two dimensions** (e.g. 7×3, 1×8): a reasonable caller expects the mip chain's
  per-level extents to follow `max(1, base >> i)` exactly, with no level ever reaching 0 in either
  dimension. Task 1's tests cover this directly from the spec's worked examples.
- **A 1×1 texture requested with `Generate`**: a reasonable caller expects this to succeed with one
  level and no blit attempted (not an error, not a spurious capability requirement) — Task 1 and
  Task 3 both cover this (planner emits zero blits; capability check is skipped for a single-level
  result).
- **Partial upload failure (mid-chain blit or final submission failure)**: a reasonable caller
  expects no `TextureId` to resolve, no descriptor-store entry to exist, and no leaked GPU resources
  — not a texture that silently resolves to white, and not a double-free. Task 4 covers this via the
  existing `StoredTexture`/`create_texture_with_options` rollback contract (spec 012's shape).
- **A device lacking blit/filter-linear support asked for `Generate` on a >1×1 texture**: a
  reasonable caller expects a clear, retryable error naming the missing capability — not a hang, not
  a validation-layer error surfacing instead of a clean Rust `Result`, and not a silent downgrade to
  one level. Task 2's pure capability helper and Task 3's wiring both cover this with synthetic
  negative capability combinations (no driver needed to test the decision logic).
- **The font atlas and white-default paths must not regress**: a reasonable caller of the existing
  `create_texture`/text-overlay API expects their one-level, nearest-filtered images to keep working
  exactly as before once the multi-level code path exists alongside them. This is structurally
  guaranteed, not just tested: Task 3 keeps `new_r8`'s call site passing a hardcoded
  `MipPolicy::OneLevel` directly (never going through `select_mip_policy`/a capability query at
  all), and `levels_for_policy(MipPolicy::OneLevel) == 1` is covered by Task 3's own test. The
  white-default texture path goes through `create_texture_with_options(..., TextureOptions::default())`
  exactly like every pre-existing `create_texture` caller, so it inherits the same guarantee.

---

## Task 1: Mip-extent planning and command-recording helpers (pure, no GPU)

**Files:**
- Create: `src/renderer/mip.rs`
- Modify: `src/renderer/mod.rs:1-20` (module declaration, `mod mip;`)

**Interfaces:**
- Produces (consumed by Tasks 2, 3):
  - `pub(crate) fn mip_level_count(width: u32, height: u32) -> Result<u32>` — `floor(log2(max(width,
    height))) + 1` via integer bit operations (`u32::BITS - 1 - max(width,height).leading_zeros()`,
    `+ 1`); returns `VulkanError::ValidationError` if `width == 0 || height == 0`.
  - `pub(crate) fn mip_extent(base_width: u32, base_height: u32, level: u32) -> (u32, u32)` —
    `(max(1, base_width >> level), max(1, base_height >> level))`.
  - `pub(crate) struct BlitStep { pub src_level: u32, pub dst_level: u32, pub src_extent: (u32, u32),
    pub dst_extent: (u32, u32) }`
  - `pub(crate) fn plan_blits(base_width: u32, base_height: u32, level_count: u32) -> Vec<BlitStep>`
    — one `BlitStep` per level `1..level_count`, `src_level = i - 1`, `dst_level = i`, extents from
    `mip_extent`. Returns an empty `Vec` when `level_count <= 1`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn mip_level_count_matches_spec_worked_examples() {
    assert_eq!(mip_level_count(1, 1).unwrap(), 1);
    assert_eq!(mip_level_count(8, 4).unwrap(), 4);  // 8x4,4x2,2x1,1x1
    assert_eq!(mip_level_count(7, 3).unwrap(), 3);  // 7x3,3x1,1x1
    assert_eq!(mip_level_count(1, 8).unwrap(), 4);  // 1x8,1x4,1x2,1x1
}

#[test]
fn mip_level_count_rejects_zero_dimensions() {
    assert!(mip_level_count(0, 4).is_err());
    assert!(mip_level_count(4, 0).is_err());
}

#[test]
fn mip_extent_never_reaches_zero_in_either_dimension() {
    assert_eq!(mip_extent(8, 4, 0), (8, 4));
    assert_eq!(mip_extent(8, 4, 1), (4, 2));
    assert_eq!(mip_extent(8, 4, 2), (2, 1));
    assert_eq!(mip_extent(8, 4, 3), (1, 1));
    assert_eq!(mip_extent(7, 3, 2), (1, 1));
}

#[test]
fn plan_blits_is_empty_for_one_level() {
    assert!(plan_blits(1, 1, 1).is_empty());
    assert!(plan_blits(8, 4, 1).is_empty());
}

#[test]
fn plan_blits_emits_one_step_per_level_in_order_with_correct_extents() {
    let steps = plan_blits(8, 4, 4);
    assert_eq!(steps.len(), 3);
    assert_eq!((steps[0].src_level, steps[0].dst_level), (0, 1));
    assert_eq!((steps[0].src_extent, steps[0].dst_extent), ((8, 4), (4, 2)));
    assert_eq!((steps[1].src_level, steps[1].dst_level), (1, 2));
    assert_eq!((steps[1].src_extent, steps[1].dst_extent), ((4, 2), (2, 1)));
    assert_eq!((steps[2].src_level, steps[2].dst_level), (2, 3));
    assert_eq!((steps[2].src_extent, steps[2].dst_extent), ((2, 1), (1, 1)));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::mip`
Expected: compile error — `mip` module/functions don't exist yet.

- [ ] **Step 3: Implement `mip.rs`**

Pure functions and the `BlitStep` struct as specified above. No `ash`/GPU types beyond what's
already needed for extents (plain `u32` tuples are enough here — `vk::Extent2D` conversion happens
in Task 3, where the recording code lives).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib renderer::mip`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/mip.rs src/renderer/mod.rs
git commit -m "feat(renderer): pure mip-level-count and blit-plan helpers"
```

---

## Task 2: Capability query and the pure accept/reject helper

**Files:**
- Modify: `src/renderer/mip.rs` (add capability-decision helper)

**Interfaces:**
- Consumes: nothing from Task 1 beyond what's already in `mip.rs`.
- Produces (consumed by Task 3):
  - `pub(crate) enum MipPolicy { OneLevel, Generate { level_count: u32 } }`
  - `pub(crate) fn select_mip_policy(requested: crate::renderer::texture::Mipmaps, width: u32,
    height: u32, format_features: ash::vk::FormatFeatureFlags, image_format_properties:
    Option<ash::vk::ImageFormatProperties>) -> Result<MipPolicy>` — pure decision function:
    - `Mipmaps::Disabled` → always `Ok(MipPolicy::OneLevel)`, no capability check at all (font/white
      paths never acquire the multi-level requirement).
    - `Mipmaps::Generate` with `mip_level_count(width, height)? == 1` (i.e. a 1×1 texture) →
      `Ok(MipPolicy::Generate { level_count: 1 })`, no capability check (nothing to blit).
    - `Mipmaps::Generate` with `level_count > 1` → require `format_features` to contain
      `BLIT_SRC | BLIT_DST | SAMPLED_IMAGE_FILTER_LINEAR | SAMPLED_IMAGE`; require
      `image_format_properties` to be `Some(_)` (caller passes `None` when the
      `get_physical_device_image_format_properties` query itself returned
      `VK_ERROR_FORMAT_NOT_SUPPORTED`) and its `max_mip_levels >= level_count` and its `max_extent`
      to cover `(width, height)`. Missing any of these →
      `Err(VulkanError::ValidationError("..."))` naming which capability is missing. All present →
      `Ok(MipPolicy::Generate { level_count })`.

(Note: this task does not call any real Vulkan device/instance function — it takes already-queried
`vk::FormatFeatureFlags`/`vk::ImageFormatProperties` as plain data, per the spec's "pure helper fed by
queried properties" requirement. Task 3 wires the real queries in.)

- [ ] **Step 1: Write the failing tests**

```rust
use ash::vk;

const FULL_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
    vk::FormatFeatureFlags::BLIT_SRC.as_raw()
        | vk::FormatFeatureFlags::BLIT_DST.as_raw()
        | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR.as_raw()
        | vk::FormatFeatureFlags::SAMPLED_IMAGE.as_raw(),
);

fn full_image_format_properties(level_count: u32) -> vk::ImageFormatProperties {
    vk::ImageFormatProperties {
        max_extent: vk::Extent3D { width: 4096, height: 4096, depth: 1 },
        max_mip_levels: level_count,
        max_array_layers: 1,
        sample_counts: vk::SampleCountFlags::TYPE_1,
        max_resource_size: u64::MAX,
    }
}

#[test]
fn disabled_always_succeeds_as_one_level_without_a_capability_check() {
    let policy = select_mip_policy(Mipmaps::Disabled, 256, 256,
        vk::FormatFeatureFlags::empty(), None).unwrap();
    assert!(matches!(policy, MipPolicy::OneLevel));
}

#[test]
fn generate_on_1x1_succeeds_without_a_capability_check() {
    let policy = select_mip_policy(Mipmaps::Generate, 1, 1,
        vk::FormatFeatureFlags::empty(), None).unwrap();
    assert!(matches!(policy, MipPolicy::Generate { level_count: 1 }));
}

#[test]
fn generate_with_full_support_succeeds_with_expected_level_count() {
    let policy = select_mip_policy(Mipmaps::Generate, 256, 256,
        FULL_FEATURES, Some(full_image_format_properties(9))).unwrap();
    assert!(matches!(policy, MipPolicy::Generate { level_count: 9 }));
}

#[test]
fn generate_rejects_each_missing_required_format_feature_individually() {
    for missing in [
        vk::FormatFeatureFlags::BLIT_SRC,
        vk::FormatFeatureFlags::BLIT_DST,
        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        vk::FormatFeatureFlags::SAMPLED_IMAGE,
    ] {
        let features = vk::FormatFeatureFlags::from_raw(FULL_FEATURES.as_raw() & !missing.as_raw());
        assert!(select_mip_policy(Mipmaps::Generate, 256, 256, features,
            Some(full_image_format_properties(9))).is_err());
    }
}

#[test]
fn generate_rejects_unsupported_image_format_properties_query() {
    assert!(select_mip_policy(Mipmaps::Generate, 256, 256, FULL_FEATURES, None).is_err());
}

#[test]
fn generate_rejects_device_mip_level_cap_below_required_count() {
    assert!(select_mip_policy(Mipmaps::Generate, 256, 256, FULL_FEATURES,
        Some(full_image_format_properties(4))).is_err()); // needs 9, device caps at 4
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::mip`
Expected: compile error — `MipPolicy`/`select_mip_policy`/`Mipmaps` don't exist yet (`Mipmaps` is
Task 3's; stub it as a local test-only `use` against a `pub(crate) enum Mipmaps` you add to `mip.rs`
in this task and re-export/re-point from `texture.rs` in Task 3 — see Task 3's Interfaces note).

- [ ] **Step 3: Implement**

Add `MipPolicy`, `select_mip_policy` to `mip.rs` as specified. Define `Mipmaps` here too (`#[derive(
Copy, Clone, Debug, PartialEq, Eq)] pub(crate) enum Mipmaps { Disabled, Generate }`) — Task 3 moves
the *public* `Mipmaps`/`TextureOptions` types into `texture.rs` and re-exports from `lib.rs`; keeping
the enum's definition in `mip.rs` for this task avoids a forward reference that doesn't exist yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib renderer::mip`
Expected: all tests from Task 1 and Task 2 PASS (12 total in the module).

- [ ] **Step 5: Commit**

```bash
git add src/renderer/mip.rs
git commit -m "feat(renderer): pure mip capability accept/reject decision"
```

---

## Task 3: Wire mip policy into `SampledImage`/`StoredTexture`; public `TextureOptions` API

**Files:**
- Modify: `src/renderer/image.rs` (full rewrite of `SampledImage::new`/`new_rgba8_srgb` to take a
  policy and record the blit chain; `new_r8` unchanged in behavior, passes `MipPolicy::OneLevel`)
- Modify: `src/renderer/texture.rs` (public `Mipmaps`, `TextureOptions`; `StoredTexture::new` gains
  an `options: TextureOptions` parameter and performs the capability query)
- Modify: `src/renderer/mip.rs` (remove the task-2 placeholder `Mipmaps`; re-point `select_mip_policy`
  at `texture::Mipmaps`)
- Modify: `src/renderer/mod.rs` (`create_texture` delegates to new `create_texture_with_options`)
- Modify: `src/renderer/headless.rs` (add `create_texture`/`create_texture_with_options`, mirroring
  `mod.rs` — needed by Task 5's GPU tests; `HeadlessRenderer` currently has no texture-creation method
  at all, only the white-default texture built at construction)
- Modify: `src/lib.rs` (re-export `Mipmaps`, `TextureOptions`)

**Interfaces:**
- Consumes: `mip::{MipPolicy, select_mip_policy, mip_level_count, plan_blits, BlitStep}` (Tasks 1-2).
- Produces (consumed by Tasks 4-5):
  - `pub struct Mipmaps` (enum, as in the spec's Public API section) and `pub struct TextureOptions {
    pub mipmaps: Mipmaps }` with `impl Default for TextureOptions` (`Mipmaps::Disabled`), in
    `texture.rs`, re-exported from `lib.rs`.
  - `VulkanRenderer::create_texture_with_options(&mut self, rgba8: &[u8], width: u32, height: u32,
    options: TextureOptions) -> Result<TextureId>` and `HeadlessRenderer::create_texture_with_options`
    with the identical signature (minus `pub`).
  - `VulkanRenderer::create_texture`/`HeadlessRenderer::create_texture` keep their exact current
    signatures, now one-line delegations to `create_texture_with_options(..., TextureOptions::default())`.
  - `StoredTexture::new(device, allocator, instance: &VulkanInstance, layout, rgba8, width, height,
    options: TextureOptions) -> Result<Self>` — the `instance` parameter is new (needed for the
    format/image-format-properties queries); both call sites (white-default construction, real
    `create_texture_with_options`) are updated in this same task so the crate keeps compiling.

- [ ] **Step 1: Write the failing tests** (in `image.rs` and `texture.rs`, alongside existing tests)

```rust
// image.rs
#[test]
fn one_level_policy_allocates_exactly_one_level_and_plans_no_blits() {
    // Exercise the pure planning path `SampledImage::new` delegates to before any GPU call:
    // level_count_for(MipPolicy::OneLevel) == 1, and plan_blits(.., .., 1).is_empty() (already
    // covered in Task 1) — this test asserts `SampledImage`'s own glue function, e.g.
    // `fn levels_for_policy(policy: MipPolicy) -> u32`, returns 1 for `MipPolicy::OneLevel` and for
    // `MipPolicy::Generate { level_count: 1 }`, and returns `level_count` for `Generate { level_count: n }`.
    assert_eq!(levels_for_policy(MipPolicy::OneLevel), 1);
    assert_eq!(levels_for_policy(MipPolicy::Generate { level_count: 1 }), 1);
    assert_eq!(levels_for_policy(MipPolicy::Generate { level_count: 5 }), 5);
}

// texture.rs
#[test]
fn texture_options_default_is_disabled() {
    assert_eq!(TextureOptions::default().mipmaps, Mipmaps::Disabled);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::image renderer::texture`
Expected: compile errors — `levels_for_policy`/`TextureOptions`/`Mipmaps` (as re-pointed) don't exist
in these modules yet.

- [ ] **Step 3: Implement**

- `texture.rs`: move/define `Mipmaps`/`TextureOptions` as public types (spec's exact shape). Add
  `levels_for_policy` in `image.rs` (or inline where used) as the trivial match shown above.
- `image.rs`: `SampledImage::new` gains a `policy: MipPolicy` parameter. Image creation now uses
  `mip_levels(levels_for_policy(policy))`; usage flags add `TRANSFER_SRC` only when
  `levels_for_policy(policy) > 1`. The upload-command closure passed to `upload::submit_once` records,
  in order: (a) a barrier moving **all** allocated levels UNDEFINED → TRANSFER_DST_OPTIMAL, (b) the
  existing base-level buffer-to-image copy (level 0 only — unchanged), (c) for each `BlitStep` from
  `mip::plan_blits(width, height, levels_for_policy(policy))` in order: barrier `src_level`
  TRANSFER_DST_OPTIMAL → TRANSFER_SRC_OPTIMAL, `cmd_blit_image` from `src_level`/`src_extent` to
  `dst_level`/`dst_extent` with `vk::Filter::LINEAR`, barrier `src_level` TRANSFER_SRC_OPTIMAL →
  SHADER_READ_ONLY_OPTIMAL, (d) a final barrier moving the last level (`levels_for_policy(policy) -
  1`) TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (this also covers the one-level case, where
  "last level" is level 0 and there were no blits). View/sampler: `level_count` in the view's
  subresource range becomes `levels_for_policy(policy)`; sampler gets `mipmap_mode: LINEAR` (vs the
  current unconditional `NEAREST`) only for the RGBA8 path (`new_rgba8_srgb` keeps passing its own
  mipmap-mode choice through to `new`, same pattern `filter` already uses), `min_lod: 0.0`, `max_lod:
  (levels_for_policy(policy) - 1) as f32`. `new_r8`'s call site passes `MipPolicy::OneLevel` and its
  existing `NEAREST`/`NEAREST` filter/mipmap-mode — no behavior change.
- `texture.rs`: `StoredTexture::new` gains `instance: &VulkanInstance` and `options: TextureOptions`
  params. Before calling `SampledImage::new_rgba8_srgb`, query
  `instance.instance.get_physical_device_format_properties(device.physical_device,
  vk::Format::R8G8B8A8_SRGB)` for `format_features`, and (only when `options.mipmaps ==
  Mipmaps::Generate` and the texture is larger than 1×1 — cheap short-circuit, matches Task 2's
  "1×1 needs no check" rule) `instance.instance.get_physical_device_image_format_properties(...)`
  with the real `width`/`height`/computed level count/usage flags, mapping
  `vk::Result::ERROR_FORMAT_NOT_SUPPORTED` to `None` and any other error to
  `Err(VulkanError::from(...))`; feed both into `mip::select_mip_policy`, and pass the resulting
  `MipPolicy` into `SampledImage::new_rgba8_srgb`.
- `mod.rs`: `create_texture_with_options` is `create_texture`'s current body plus the `options`
  param and the new `instance` argument to `StoredTexture::new` (`&self.instance`); `create_texture`
  becomes `self.create_texture_with_options(rgba8, width, height, TextureOptions::default())`. Same
  split for the white-texture construction call site (pass `TextureOptions::default()`, i.e.
  `Mipmaps::Disabled` — unchanged behavior, just the new required arguments).
- `headless.rs`: add `create_texture`/`create_texture_with_options` as `pub(crate) fn`s mirroring
  `mod.rs`'s, using `self.instance`/`self.device`/`self.allocator`/`self.texture_set_layout`, pushing
  onto new `pub(crate)` fields `textures: Vec<StoredTexture>` and `texture_sets: Vec<vk::DescriptorSet>`
  (mirrors `mod.rs`'s field names exactly) added to the `HeadlessRenderer` struct and initialized as
  empty `Vec::new()` in its constructor. Update the white-texture construction call site here too
  (new `instance`/`options` args).
- `lib.rs`: add `Mipmaps`, `TextureOptions` to the `pub use renderer::texture::{...}` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib renderer::image renderer::texture renderer::mip`
Expected: all tests pass, including Tasks 1-2's unchanged.

Run: `cargo build --all-targets && cargo test --all-targets`
Expected: full crate builds; all pre-existing tests still pass (this task touches two real call
sites of `StoredTexture::new` in `mod.rs` and introduces two new ones in `headless.rs` — confirm no
caller was missed, per the Global Constraints "no second caller breaks" rule).

- [ ] **Step 5: Commit**

```bash
git add src/renderer/image.rs src/renderer/texture.rs src/renderer/mip.rs src/renderer/mod.rs \
  src/renderer/headless.rs src/lib.rs
git commit -m "feat(renderer): generate full mip chains for opted-in RGBA8 textures"
```

---

## Task 4: Failure-path and ownership tests (no new production code expected)

**Files:**
- Modify: `src/renderer/texture.rs` (tests only)
- Modify: `src/renderer/headless.rs` (tests only, `#[cfg(test)]` module)

**Interfaces:**
- Consumes: `create_texture_with_options` (Task 3), the existing `guard`/rollback machinery already
  used by `create_texture`/`create_mesh` (spec 012's shape — unchanged by this plan).
- Produces: nothing new; this task is pure test coverage proving Task 3's wiring honors spec 012's
  existing contract for the new code paths. If a gap is found, fix it in `texture.rs`/`mod.rs`/
  `headless.rs` as a small addendum to this task's commit (not a new task) — the plan does not
  pre-specify a fix because none is expected if Task 3 reused the existing guard pattern correctly.

- [ ] **Step 1: Write the failing tests**

```rust
// texture.rs — capability rejection is reachable from the public options path, not just
// select_mip_policy in isolation:
#[test]
fn create_texture_with_options_generate_on_oversized_request_rejects_before_any_texture_id_exists() {
    // Build a HeadlessRenderer, request Generate on a texture larger than the device's
    // max_image_dimension_2d-style limit is impractical to synthesize without a real device, so
    // this test instead confirms the *error path* end-to-end using a deliberately mismatched
    // rgba8.len() (reuses SampledImage's existing validate_pixels check, which now runs before any
    // mip-specific GPU work) -- confirming no TextureId/descriptor entry is published and the
    // renderer is not left terminal for a non-terminal validation error.
}

#[test]
fn successful_generate_call_publishes_exactly_one_texture_id_and_one_descriptor_set() {
    // Happy-path on a real (ignored, GPU) HeadlessRenderer: create_texture_with_options with
    // Generate on a small (e.g. 4x4) texture succeeds; texture_sets.len() and textures.len() both
    // grew by exactly 1; the returned TextureId resolves via resolve_texture_set.
}
```

(These two illustrate the shape; the implementer writes the exact assertions against the real
`HeadlessRenderer` test helpers already used by `mesh.rs`'s/`texture.rs`'s existing failure-path
tests — follow those existing tests' style for constructing a renderer and triggering the error.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::texture renderer::headless -- --ignored`
Expected: the happy-path GPU test fails (or doesn't exist) before Task 3's `create_texture_with_options`
plumbing is exercised this way; confirm failure is for the right reason, not a typo.

- [ ] **Step 3: Implement** (tests only; see note above about addendum fixes if a gap surfaces)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --all-targets` (CPU-only subset) and, on a machine with a driver,
`cargo test --lib renderer::texture renderer::headless -- --ignored`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/texture.rs src/renderer/headless.rs
git commit -m "test(renderer): cover mip-generation failure paths and ownership contract"
```

---

## Task 5: GPU acceptance — readback fixture, golden-image scene tests, cube example demo

**Files:**
- Create: `src/renderer/mip_readback.rs` (test-only; `#[cfg(test)]` at the module-declaration site)
- Modify: `src/renderer/mod.rs` (`mod mip_readback;` under `#[cfg(test)]`)
- Modify: `src/renderer/headless.rs` (tests: two golden-image scene tests)
- Create: `testdata/golden/mipmap_disabled_far.png`, `testdata/golden/mipmap_generate_far.png`,
  `testdata/golden/mipmap_generate_near.png` (generated on real hardware per the project's
  `M2S2_UPDATE_GOLDEN=1` workflow — not committed by the implementer; see Step 5 note)
- Modify: `examples/cube.rs` (replace the 2×2 checker with a 256×256 procedural checker, 4×4-texel
  checks; add a labeled Disabled-vs-Generate comparison per spec's "Interactions and demonstration")

**Interfaces:**
- Consumes: `HeadlessRenderer::create_texture_with_options` (Task 3), `golden::assert_matches_golden`
  (spec 019's existing infra, tolerance literal `2` per the project's convention), `upload::submit_once`
  (existing).
- Produces: nothing consumed by a later task (this is the plan's final task).

- [ ] **Step 1: Write the failing tests**

```rust
// mip_readback.rs — new test-only helper, mirrors upload.rs's one-time-submission shape:
#[cfg(test)]
pub(crate) fn read_back_mip_level(
    device: &VulkanDevice, allocator: &mut Allocator, image: vk::Image,
    level: u32, width: u32, height: u32,
) -> Result<Vec<u8>> {
    // Transition `level` to TRANSFER_SRC_OPTIMAL, copy_image_to_buffer into a new host-visible
    // staging Buffer sized width*height*4, transition back to SHADER_READ_ONLY_OPTIMAL (the image
    // stays usable afterward — test-only, torn down normally by the caller's HeadlessRenderer
    // Drop), map and return the bytes. One `upload::submit_once` call.
}

// headless.rs, #[cfg(test)] mod tests, #[ignore]:
#[test]
fn generated_mip_level_1_of_quadrant_texture_matches_each_quadrant_exactly() {
    // 16x16 RGBA8 sRGB, four solid 8x8 quadrants (red/green/blue/white), Generate. Level 1 is 8x8:
    // read back via read_back_mip_level, assert each of the four 4x4 sub-blocks (one per original
    // quadrant) is within tolerance 2/255 per channel of that quadrant's exact color -- per the
    // spec's resolved decision, a box filter over a constant block equals that constant regardless
    // of linear-vs-sRGB averaging, so this needs no filtering-tolerance judgment call.
}

#[test]
fn generated_mip_chain_has_expected_level_count_and_shrinking_dimensions() {
    // Same 16x16 fixture: levels 0..5 (16,8,4,2,1) all readable, each half the prior (rounding per
    // mip_extent), confirming the chain is real GPU content at every level, not just level 1.
}

// headless.rs, golden-image scene tests (reuse spec 019's pattern exactly):
#[test]
fn mipmap_disabled_vs_generate_differ_at_far_camera_distance() {
    // Render the same checker-textured quad/cube at a fixed far distance, once with Disabled, once
    // with Generate; assert the two pixel buffers are NOT within tolerance of each other (proves
    // mipmapping visibly changes the far-distance render) -- then separately assert_matches_golden
    // each against its own committed baseline (mipmap_disabled_far.png / mipmap_generate_far.png).
}

#[test]
fn mipmap_generate_at_near_camera_distance_matches_golden() {
    // Same geometry, near camera, Generate only -- assert_matches_golden against
    // mipmap_generate_near.png, confirming close-up sampling isn't visibly degraded.
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::mip_readback renderer::headless -- --ignored`
Expected: compile errors (helper/tests don't exist), or `missing golden image` errors once they
compile — both are the expected "fails for the right reason" state before goldens exist.

- [ ] **Step 3: Implement**

Write `read_back_mip_level` and the four tests above. For the golden-scene tests, follow the exact
pattern of an existing `headless.rs` golden test (e.g. the point-light or multi-mesh ones) for
camera/object/render-call shape; the only new element is varying `TextureOptions.mipmaps` and the
far/near camera position.

Then update `examples/cube.rs`: replace the `checker` helper's 2×2 call sites (lines ~173-186) with
a 256×256 procedural checker (4×4-texel checks, per the spec), and extend the scene so one object
uses `create_texture_with_options(..., TextureOptions { mipmaps: Mipmaps::Generate })` and a second,
clearly labeled (via the existing `Diagnostics`/text-overlay path, not a new UI) uses `Disabled`
with the same checker pattern, so moving the free-fly camera from near to far demonstrates the
difference. This is a manual/visual deliverable, not unit-tested — `cargo build --examples` catches
drift; running it is the user's GPU-verification step (Task 5's completion contract below does not
require running the example, only building it).

- [ ] **Step 4: Run tests to verify they pass**

CPU-only (no driver): `cargo build --all-targets && cargo test --all-targets` — confirms everything
compiles, the ignored tests are skipped cleanly, and `cargo build --examples` picks up the cube
changes.

GPU (hands off to the user, same pattern as specs 015/016/018/019): on a machine with a Vulkan
driver, run `M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless -- --ignored
mipmap_disabled_vs_generate_differ_at_far_camera_distance
mipmap_generate_at_near_camera_distance_matches_golden` once to create the three new goldens, then
re-run the same tests *without* the env var to confirm they now pass against what was just written,
then run the **full** `cargo test --lib renderer::headless -- --ignored` (every pre-existing ignored
test, not just this task's new ones) and confirm every pre-existing golden PNG is still
byte-identical via `git diff --quiet testdata/golden/<file>` for each one already in the repo before
this task — the project's established zero-regression check (see spec 016's Outcome for the exact
same verification shape applied there).

- [ ] **Step 5: Commit**

Two commits, matching the project's established code/goldens split:

```bash
git add src/renderer/mip_readback.rs src/renderer/mod.rs src/renderer/headless.rs examples/cube.rs
git commit -m "test(renderer): GPU mip-chain readback and golden-image acceptance tests"
# After the user generates goldens on real hardware and the zero-regression check above passes:
git add testdata/golden/mipmap_disabled_far.png testdata/golden/mipmap_generate_far.png \
  testdata/golden/mipmap_generate_near.png
git commit -m "test(golden): add mipmap disabled/generate near/far acceptance images"
```
