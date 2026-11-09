# Point Lighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add up to eight positioned, colored, finite-range point lights that share the existing
Blinn–Phong diffuse/specular response with the directional light, verified by real automated
pixel-comparison GPU tests (spec 019's headless/golden-image infrastructure).

**Architecture:** Extend the existing single-directional-light `LightUniforms` block into one
`SceneLightUniforms` block carrying the directional fields plus a fixed 8-element point-light array
and a count. The CPU-side `Light` wrapper, `VulkanRenderer`, `HeadlessRenderer`, and `command.rs`'s
per-frame upload all grow to carry the point-light list alongside the existing directional light. The
fragment shader factors its inlined Blinn–Phong math into one reusable function so the directional
light and every point light call the identical code. No new descriptor, no new per-light allocation —
one larger UBO, one bind.

**Tech Stack:** Rust, `ash` (Vulkan 1.3 dynamic rendering), `bytemuck` (Pod/Zeroable uniform mirrors),
GLSL (`glslc`/`spirv-val`), `m2s2-math`, spec 019's `HeadlessRenderer`/`assert_matches_golden` for GPU
acceptance.

**Spec:** [specs/018-point-lighting.md](../../../specs/018-point-lighting.md)

## Global Constraints

- `MAX_POINT_LIGHTS = 8`. `PointLight { position: Vec3, color: Vec3, range: f32 }` — position finite,
  color finite/nonnegative, range finite and `> 0`.
- `PointLightUniforms` (std140, 32 bytes): `position: Vec3` @0, `range: f32` @12, `color: Vec3` @16,
  `_pad: f32` @28.
- `SceneLightUniforms` (std140, 320 bytes total): existing directional fields unchanged at
  `direction`@0/`color`@16/`ambient`@32 (each vec3+pad, 48 bytes), `point_count: u32` @48, 12 bytes
  padding to 64, then `points: [PointLightUniforms; 8]` @64 (256 bytes) = 320 bytes total.
- Falloff (fragment shader, evaluated only past the degeneracy guard): `t = 1 - d/range; attenuation =
  t*t`; zero when `d >= range`. Degeneracy guard (squared distance `<= 1e-12`) runs *before* computing
  `d`/`L`/`t` and contributes zero immediately — it is not a branch of the falloff formula.
- `set_point_lights` validates the *entire* list before mutating any renderer state: reject more than
  eight, any non-finite position, non-finite/negative color component, or non-finite/nonpositive range.
  Rejection leaves the previous list completely unchanged (no partial truncation, no partial apply).
  Default list is empty.
- No new descriptor set, descriptor binding, or per-light Vulkan allocation. Set 2 binding 0's UBO
  simply grows from 48 to 320 bytes; the existing bind covers it.
- No uniform-buffer-range device-limit check exists yet (only `max_push_constants_size` is tracked
  today). Add a new `max_uniform_buffer_range: u32` field to `VulkanDevice` (both `new` and
  `new_headless`) and a new `validate_light_uniform_limit(limit: u32) -> Result<()>` in `light.rs`,
  shaped like `pipeline.rs`'s existing `validate_push_limit`, called from `Light::new`.
- `HeadlessRenderer::render_frame` currently hardcodes `DirectionalLight::default()` and accepts no
  point lights at all — its signature must grow `directional_light: &DirectionalLight` and
  `point_lights: &[PointLight]` parameters. This breaks the existing (test-only, non-public) call
  sites in `headless.rs`'s own test module; update them to pass `&DirectionalLight::default()` and
  `&[]` to preserve their current behavior.
- `triangle.frag` has no reusable specular/diffuse function today — directional shading is inlined
  directly in `main()`. Extract a new `vec3 blinnPhongSpecular(vec3 N, vec3 L, vec3 V, vec3
  specularColor, float shininess)` function; the directional light and every point light call it. The
  camera-to-surface vector `V` is computed once per fragment and shared across all light calls.
- `set_point_lights` is purely additive — `render_frame`/`render_objects`/`render_objects_with_text`
  signatures do not change (unlike spec 013's `Camera` migration).
- Quality gates zero-warning at every commit: `cargo fmt --all -- --check`, `cargo clippy
  --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test
  --all-targets`. Shader changes additionally require `glslc` + `spirv-val` clean on both recompiled
  `.spv` files (manual verification step, not an automated test — matches spec 013's precedent).
  Conventional Commits; no `Co-Authored-By`/AI-attribution trailer.
- GPU-dependent tests (Task 4) are `#[ignore]`d by default, per spec 019's established pattern; this
  plan's executor has no GPU and cannot run them. Each new golden-image test needs its own reference
  PNG generated once by the user via `M2S2_UPDATE_GOLDEN=1`, same handoff spec 019 established.

## Review Focus

- **`set_point_lights` called with exactly nine lights, or with eight valid lights followed by one
  invalid one in the same call** — the whole call must reject and leave the previous list completely
  untouched, not accept the first eight and reject the ninth, and not partially apply before hitting
  the invalid element. *Test in Task 2.*
- **A point light positioned exactly at (or within `1e-12` squared distance of) the fragment being
  shaded** — must contribute zero without ever normalizing a zero-length direction (NaN/Inf
  propagation into `outColor` would be silent and only visible as a corrupted pixel). *Test in Task 4
  via a golden-image scenario; the degeneracy guard itself is CPU-reviewed by inspection in Task 3,
  since GLSL branches aren't independently unit-testable without a GPU.*
- **`point_count` of zero** — must produce a result visually and numerically identical to spec 013's
  baseline (zero point-light shader accesses, not just zero contribution from a zero-filled array).
  *Test in Task 4 (`zero points matches baseline` golden scenario).*
- **Ambient must not scale with point-light count** — summing N point contributions must never also
  multiply the single ambient term by N. A reviewer should be able to point at the exact shader line
  that adds ambient exactly once, outside any per-light loop. *Verified by code inspection in Task 3's
  review (not independently GPU-testable in isolation from the golden scenarios), and indirectly by
  Task 4's "colored lights add" and "all eight lights" golden scenarios not blowing out brightness.*
- **The 320-byte `SceneLightUniforms` buffer binds correctly even though `triangle.frag`'s old-shaped
  `LightUniforms` block only declares the first 48 bytes** — Vulkan permits a bound UBO range larger
  than what the shader block declares (same pattern spec 013 already relied on for `CameraUniforms`
  growing past what the vertex shader redeclares), but this is worth a reviewer's explicit sign-off
  since it's easy to assume a shader/CPU struct mismatch is always an error. *Reviewed by inspection in
  Task 1 (layout tests) and Task 3 (shader change); no dedicated test needed beyond the existing
  layout-size assertions.*

---

### Task 1: `PointLight` type, GPU layout, and device uniform-range limit

**Files:**
- Modify: `src/renderer/light.rs`
- Modify: `src/renderer/device.rs`
- Test: inline `#[cfg(test)] mod tests` in both files above

**Interfaces:**
- Produces: `pub struct PointLight { pub position: Vec3, pub color: Vec3, pub range: f32 }` (re-export
  deferred to Task 2, once `mod.rs` consumes it); `pub const MAX_POINT_LIGHTS: usize = 8`;
  `pub(crate) fn validate_point_lights(lights: &[PointLight]) -> Result<()>`; `pub(crate) struct
  PointLightUniforms` (repr(C), Pod, Zeroable); `pub(crate) struct SceneLightUniforms` (repr(C), Pod,
  Zeroable) replacing `LightUniforms` as `Light`'s payload type; `Light::update(&mut self, frame:
  usize, directional: &DirectionalLight, points: &[PointLight]) -> Result<()>` (signature grows from
  today's `update(&mut self, frame: usize, light: &DirectionalLight)`); `VulkanDevice`'s new
  `max_uniform_buffer_range: u32` field (set in both `new` and `new_headless`); `pub(crate) fn
  validate_light_uniform_limit(limit: u32) -> Result<()>` in `light.rs`, called inside `Light::new`.
- Consumes: nothing new from other tasks — this task only touches `light.rs`/`device.rs`.

- [ ] **Step 1: Write failing tests for `PointLight` validation**

In `light.rs`'s test module, add tests covering: an empty list and a list of exactly 8 valid lights
both accepted; a list of 9 rejected; each of non-finite position component, non-finite color
component, negative color component, non-finite range, zero range, and negative range rejected
individually; and — matching the Review Focus item above — a call with 8 valid lights followed by 1
invalid light rejected as a whole (not silently truncated to 8).

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cargo test --lib renderer::light`
Expected: FAIL — `validate_point_lights` not defined.

- [ ] **Step 3: Implement `PointLight` and `validate_point_lights`**

Add `PointLight` next to `DirectionalLight` in `light.rs`, with the same doc-comment density as
`DirectionalLight`'s fields. `validate_point_lights` checks the whole slice up front (length first,
then every element) and returns on the first violation without mutating anything — this function has
no state to mutate itself; Task 2's `set_point_lights` is what must apply the "leave previous state
untouched on rejection" guarantee, by calling this validator before touching its own stored list.

- [ ] **Step 4: Run tests, confirm they pass**

Run: `cargo test --lib renderer::light`
Expected: PASS.

- [ ] **Step 5: Write failing tests for `PointLightUniforms`/`SceneLightUniforms` layout**

Add tests asserting: `size_of::<PointLightUniforms>() == 32` with `offset_of!` for `position`(0),
`range`(12), `color`(16); `size_of::<SceneLightUniforms>() == 320` with `offset_of!` for
`direction`(0), `color`(16), `ambient`(32), `point_count`(48), `points`(64); a conversion test building
`SceneLightUniforms` from a `DirectionalLight` and a 3-element point-light slice, asserting
`point_count == 3`, the first 3 `points` entries match the input (with their own `_pad` zeroed), and
entries 3..8 are entirely zeroed (`bytemuck::Zeroable`'s zero, not garbage).

- [ ] **Step 6: Run tests, confirm they fail**

Run: `cargo test --lib renderer::light`
Expected: FAIL — types/conversion not defined.

- [ ] **Step 7: Implement `PointLightUniforms`, `SceneLightUniforms`, and the conversion**

Replace `LightUniforms` with `SceneLightUniforms` (keep the existing `direction`/`color`/`ambient`
fields and their offsets byte-identical — only appending new fields after them preserves the Review
Focus item about the shader's old-shaped block still reading correctly). Write a constructor (e.g.
`SceneLightUniforms::new(directional: &DirectionalLight, points: &[PointLight]) -> Self`) that builds
the full 320-byte struct, zeroing every unused point slot. `points.len()` is trusted to already be
`<= MAX_POINT_LIGHTS` here (Task 2's `set_point_lights` enforces that upstream) — a `debug_assert!` is
appropriate, not a runtime `Result`.

- [ ] **Step 8: Update `Light::update`'s signature, and patch its two call sites to keep the crate compiling**

Change `Light::update(&mut self, frame, light: &DirectionalLight)` to `Light::update(&mut self, frame,
directional: &DirectionalLight, points: &[PointLight])`, building `SceneLightUniforms` internally and
writing it via the existing `self.inner.update(frame, bytemuck::bytes_of(&scene))` call. Update
`Light::new`'s payload size from `size_of::<LightUniforms>()` to `size_of::<SceneLightUniforms>()`.

This crate must compile and its full test suite must pass at every commit (project-wide pre-commit
hook enforces `cargo clippy --all-targets`/`cargo test --all-targets`; there is no way to land a commit
that leaves the crate broken). So, as part of THIS step, patch the two call sites this signature
change breaks with a literal placeholder — do NOT do any real feature wiring here, that is Task 2's
job:
- `src/renderer/command.rs`'s `draw_frame`, the line calling `light.update(frame, directional_light)`:
  add a literal `&[]` third argument. Do not change `draw_frame`'s own parameter list — Task 2 Step 3
  adds the real `point_lights: &[PointLight]` parameter and replaces this placeholder.
- `src/renderer/headless.rs`'s `render_frame`, the line calling `self.light.update(0,
  &DirectionalLight::default())`: add a literal `&[]` third argument. Do not change `render_frame`'s
  own parameter list — Task 2 Step 4 adds the real parameters and replaces this placeholder.

Both are one-line, mechanical, behavior-preserving edits (empty point-light list is already today's
implicit behavior). `mod.rs` needs no change — it calls `draw_frame`, not `Light::update`, directly.

- [ ] **Step 9: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean, same pass/ignored counts as the branch baseline (this task adds new passing tests
in `renderer::light`/`renderer::device` and changes no runtime behavior anywhere else — the two
placeholder edits above are compile-fixes only, not feature changes).

- [ ] **Step 10: Write failing test for the device uniform-range limit**

In `device.rs`'s test module (or `light.rs`'s, wherever `validate_push_limit`'s sibling tests
currently live is not this file — place `validate_light_uniform_limit` itself in `light.rs`, next to
the types it limits, and its test alongside `validate_point_lights`'s tests): assert
`validate_light_uniform_limit(320)` is `Ok`, `validate_light_uniform_limit(319)` is `Err`, and
`validate_light_uniform_limit(16384)` (Vulkan's guaranteed minimum `maxUniformBufferRange`) is `Ok`.

- [ ] **Step 11: Run test, confirm it fails**

Run: `cargo test --lib renderer::light`
Expected: FAIL — function not defined.

- [ ] **Step 12: Implement `validate_light_uniform_limit` and the `VulkanDevice` field**

Add `pub(crate) fn validate_light_uniform_limit(limit: u32) -> Result<()>` in `light.rs`, shaped like
`pipeline.rs`'s `validate_push_limit` (compare `size_of::<SceneLightUniforms>() as u32` against
`limit`, return a `VulkanError::InitializationError` or matching variant on failure — match whatever
error variant `validate_push_limit` uses, for consistency). In `device.rs`, add
`max_uniform_buffer_range: u32` to `VulkanDevice`'s struct and populate it in both `new` and
`new_headless` from `get_physical_device_properties(...).limits.max_uniform_buffer_range`, mirroring
exactly how `max_push_constants_size` is already populated in both constructors. Call
`validate_light_uniform_limit(device.max_uniform_buffer_range)` as the first line of `Light::new`,
before constructing the `PerFrameUniform`.

- [ ] **Step 13: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean, same pass/ignored counts as Step 9 (the device-limit addition is CPU-only and
changes no runtime behavior).

- [ ] **Step 14: Commit**

```bash
git add src/renderer/light.rs src/renderer/device.rs src/renderer/command.rs src/renderer/headless.rs
git commit -m "feat(renderer): point light type, GPU layout, and uniform-range device limit"
```

---

### Task 2: Wire point lights into `VulkanRenderer`, `HeadlessRenderer`, and `command.rs`

**Files:**
- Modify: `src/renderer/mod.rs`
- Modify: `src/renderer/command.rs`
- Modify: `src/renderer/headless.rs`
- Modify: `src/lib.rs` (re-export `PointLight`, `MAX_POINT_LIGHTS`)

**Interfaces:**
- Consumes: Task 1's `PointLight`, `MAX_POINT_LIGHTS`, `validate_point_lights`, and `Light::update`'s
  new three-argument signature. Task 1 already left the crate compiling by patching
  `command.rs`'s and `headless.rs`'s `Light::update` call sites with a literal `&[]` placeholder —
  this task replaces those two placeholders with real data, it is not fixing broken code.
- Produces: `impl VulkanRenderer { pub fn set_point_lights(&mut self, lights: &[PointLight]) ->
  Result<()>; }`; `command::FrameContext::draw_frame`'s signature grows a `point_lights: &[PointLight]`
  parameter (placed next to the existing `directional_light: &DirectionalLight` parameter);
  `HeadlessRenderer::render_frame`'s signature grows `directional_light: &DirectionalLight,
  point_lights: &[PointLight]` parameters. Task 3 (shaders) and Task 4 (GPU tests) both call this
  final `render_frame` signature.

- [ ] **Step 1: Write failing tests for `VulkanRenderer::set_point_lights`' validation boundary**

This renderer method cannot be unit-tested end-to-end without a GPU (it requires a live
`VulkanRenderer`), so test the validation boundary at the level `mod.rs` actually enforces it: find
(or, if none exists, add) the lowest GPU-free seam that proves "an invalid call changes nothing." The
cleanest option: expose the check as a thin wrapper that stores into a plain `Vec<PointLight>` field
and delegates entirely to `validate_point_lights` — write the test directly against
`validate_point_lights` in Task 1's terms (already covered there) plus one or two assertions here (if
your seam allows it) confirming `set_point_lights` calls the validator before mutating
`self.point_lights`. If no non-GPU seam exists for this specific method, say so in your report instead
of fabricating a test that doesn't exercise real behavior — this will be covered end-to-end by Task
4's GPU scenarios regardless.

- [ ] **Step 2: Implement `set_point_lights` and renderer-owned point-light CPU state**

Add `point_lights: Vec<PointLight>` to `VulkanRenderer`'s fields, defaulting to empty (mirroring
`directional_light: DirectionalLight::default()`'s existing field). `set_point_lights` calls
`validate_point_lights(lights)?` first, then `self.point_lights = lights.to_vec()` only on success —
matching `set_directional_light`'s existing all-or-nothing pattern. Doc-comment it with the same
density as `set_directional_light`'s existing doc comment (takes effect next `render_*` call, rejects
with a validation error, preserves the previous list).

- [ ] **Step 3: Thread `point_lights` through `command::FrameContext::draw_frame` and its one call site**

Add `point_lights: &[PointLight]` to `draw_frame`'s parameter list (next to `directional_light`).
Inside the function, replace Task 1's `light.update(frame, directional_light, &[])` placeholder with
`light.update(frame, directional_light, point_lights)`. Update `mod.rs`'s single call site (around the
existing `self.light.as_mut().unwrap()` / `&self.directional_light` call) to pass `&self.point_lights`
too.

- [ ] **Step 4: Thread point lights through `HeadlessRenderer::render_frame`**

Change `render_frame`'s signature from `(&mut self, objects: &[Object], camera: uniform::Camera)` to
`(&mut self, objects: &[Object], camera: uniform::Camera, directional_light: &DirectionalLight,
point_lights: &[PointLight])`. Replace Task 1's `self.light.update(0, &DirectionalLight::default(),
&[])` placeholder with `self.light.update(0, directional_light, point_lights)`. Update every existing
call site in `headless.rs`'s own test module (`render_frame_is_deterministic_across_repeated_calls`'s
two calls, and the end-to-end self-test from spec 019) to pass `&DirectionalLight::default()` and
`&[]` as the two new arguments — this preserves their current tested behavior exactly; do not change
what they assert.

- [ ] **Step 5: Re-export `PointLight` and `MAX_POINT_LIGHTS` from `lib.rs`**

Add `pub use renderer::light::{DirectionalLight, PointLight, MAX_POINT_LIGHTS};` (or extend the
existing `DirectionalLight` re-export line), matching the file's existing re-export style.

- [ ] **Step 6: Build and run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean. The crate now builds end-to-end again (Task 1's `light.rs`/`device.rs` changes
plus this task's wiring resolve every compile error Task 1 deliberately left open). GPU-dependent
tests in `headless.rs` remain `#[ignore]`d, not executed.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/mod.rs src/renderer/command.rs src/renderer/headless.rs src/lib.rs
git commit -m "feat(renderer): wire point lights into VulkanRenderer, HeadlessRenderer, and the per-frame upload"
```

---

### Task 3: Shader implementation and cube example demonstration

**Files:**
- Modify: `shaders/triangle.frag`
- Modify: `examples/cube.rs`

**Interfaces:**
- Consumes: Task 2's final `SceneLightUniforms` GPU layout (already correct on the CPU side since
  Task 1; this task makes the shader actually read the point-light portion of the buffer it's already
  being sent).
- Produces: nothing new for later tasks — Task 4 renders against whatever this task's shader produces.

- [ ] **Step 1: Extend the fragment shader's light block declaration**

In `triangle.frag`, replace the `LightUniforms` block declaration with the full `SceneLightUniforms`
shape: keep `direction`/`color`/`ambient` exactly as they are (same offsets), add `uint pointCount;`
at offset 64's preceding slot (offset 48) and a fixed-size array `PointLight points[8]` (a GLSL struct
with `vec3 position; float range; vec3 color;` — std140 lays this out identically to the Rust
`PointLightUniforms`) starting at offset 64. The array bound (`8`) must equal `MAX_POINT_LIGHTS`;
write it as a GLSL `#define` or named constant rather than a bare literal, so a future change to
`MAX_POINT_LIGHTS` has an obvious shader-side counterpart to update.

- [ ] **Step 2: Extract `blinnPhongSpecular` and refactor the directional light to call it**

Pull the existing inlined specular computation (the `highlight` block in current `main()`) into a
function `vec3 blinnPhongSpecular(vec3 N, vec3 L, vec3 V, vec3 specularColor, float shininess)`
returning the highlight contribution for one light (zero if the degeneracy/front-facing guards fail).
Compute `V` (camera-to-surface, normalized, with its own degeneracy guard) once in `main()`, before any
light loop. Call `blinnPhongSpecular` once for the directional light exactly as today, producing
identical output to before this change (this refactor alone should not change any rendered pixel —
useful as your own sanity check before adding point lights).

- [ ] **Step 3: Add the point-light loop**

After the directional contribution, loop `for (uint i = 0u; i < light.pointCount; i++)`: compute the
surface-to-light vector and squared distance; if squared distance `<= MIN_SAFE_LENGTH_SQUARED` (the
existing named constant), contribute zero and continue; otherwise compute `d`, `L = (Q-P)/d`, and (if
`d < range`) `t = 1 - d/range; attenuation = t*t`, then add `attenuation * light.points[i].color *
(max(dot(N,L),0.0) * albedo.rgb + blinnPhongSpecular(N, L, V, material.specularColor,
material.shininess))` to the running shaded sum. Add ambient exactly once, outside this loop (it
already is, in the existing code — do not move it inside).

- [ ] **Step 4: Recompile and validate both shaders**

Run: `glslc shaders/triangle.vert -o shaders/triangle.vert.spv && glslc shaders/triangle.frag -o
shaders/triangle.frag.spv && spirv-val shaders/triangle.vert.spv && spirv-val shaders/triangle.frag.spv`
Expected: no errors or warnings from either `glslc` or `spirv-val`. (The vertex shader is unchanged by
this task but is recompiled for consistency with prior specs' practice of committing both `.spv`s
together.)

- [ ] **Step 5: Demonstrate point lights in the cube example**

In `examples/cube.rs`, add a call to `renderer.set_point_lights(&[...])` with two differently colored
point lights with overlapping finite ranges, positioned so both cubes fall within at least one light's
range. Move one light's position along a simple deterministic path each frame (e.g. a function of
elapsed time, same style as the existing camera/object animation in this file) to demonstrate dynamic
per-frame updates — reuse whatever time/frame-counter state the file already tracks rather than adding
a new one. You may reduce or disable the existing directional light's diffuse/ambient terms (per the
spec's suggestion) to make the point lights' effect clearly visible, or add a toggle key consistent
with the existing `KeyCode::KeyH` (highlights)/`KeyCode::F3` (diagnostics) pattern — your choice, but
keep the existing camera/highlight/diagnostic controls working unchanged.

- [ ] **Step 6: Build and run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean. (This task adds no new Rust unit tests of its own beyond what already exists —
the existing `pipeline.rs` SPIR-V-magic-number tests implicitly cover the recompiled binaries, and the
example's existing input-handling tests are unaffected.)

- [ ] **Step 7: Commit**

```bash
git add shaders/triangle.vert.spv shaders/triangle.frag.spv shaders/triangle.frag examples/cube.rs
git commit -m "feat(renderer): point-light Blinn-Phong shading in triangle.frag; demonstrate in cube example"
```

---

### Task 4: GPU golden-image acceptance tests and user handoff

**Files:**
- Modify: `src/renderer/headless.rs` (new `#[cfg(test)]` tests in the existing test module)

**Interfaces:**
- Consumes: Task 2's final `HeadlessRenderer::render_frame` signature; Task 3's completed shader (these
  tests are meaningless until both are done, since they render and compare real pixels).
- Produces: nothing — this is the plan's last task.

- [ ] **Step 1: Write the golden-image acceptance tests**

Add `#[ignore]`d tests to `headless.rs`'s test module, one per scenario below, each following the
existing `self_test_scene_matches_golden_with_zero_validation_errors` test's structure (reuse
`lit_untextured_cube()`, the same fixed camera pattern, `assert_matches_golden` at tolerance 2, and
`renderer.validation_error_count() == 0`). Each test needs a distinct golden PNG path under
`testdata/golden/`, named for its scenario (e.g. `point_light_zero_matches_baseline.png`,
`point_light_one_illuminates.png`, etc.):

- `point_light_zero_matches_baseline`: zero point lights; compare against the *same* rendered output
  spec 013's baseline self-test produces (i.e., this should be able to reuse
  `headless_self_test.png` directly as its golden, or render an identical scene and confirm it's
  pixel-equal within tolerance to that existing golden — your choice of exact mechanism, but the test
  must prove "zero points changes nothing").
- `point_light_one_illuminates`: one point light near the geometry, confirm nonzero difference from the
  zero-point baseline (a simple "is not pixel-identical to the zero-point golden" check, or a golden
  image of its own — prefer a golden of its own for consistency with the other scenarios).
- `point_light_colors_add`: two overlapping, differently colored point lights; golden image.
- `point_light_beyond_range_contributes_zero`: a point light positioned far enough outside its `range`
  that it should contribute nothing; golden image should match the zero-point baseline within
  tolerance.
- `point_light_eight_lights_and_ambient_does_not_scale`: all eight slots filled; golden image. (This is
  also where the Review Focus item about ambient-not-scaling-with-count gets its empirical check — if
  ambient were wrongly multiplied by count, this scene would be visibly blown out compared to the
  other scenarios' ambient floor.)

Each test's `#[ignore]` reason string should name its own golden PNG path, matching the existing
self-test's convention.

- [ ] **Step 2: Build (cannot run — no GPU here)**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean; every new test listed as ignored, not executed; the ignored-test count increases
by the number of scenarios added in Step 1.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/headless.rs
git commit -m "test(renderer): golden-image GPU acceptance tests for point lighting"
```

- [ ] **Step 4: Hand off to the user — generate the new golden images**

This plan's executor has no GPU and cannot complete this step. Report clearly that the user must run,
on a machine with a Vulkan driver, one `M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless --
--ignored <test_name>` invocation per new scenario from Step 1 to generate each golden PNG, then run
the same set of tests **without** `M2S2_UPDATE_GOLDEN` to confirm they all pass, and finally re-run the
full existing `#[ignore]`d suite (including spec 019's pre-existing tests) to confirm nothing
regressed. Only after the user confirms this should the golden PNGs be committed and spec 018's
Outcome section + `ROADMAP.md` be updated to reflect verified (not just implemented) status — the same
two-step handoff pattern spec 019 established.
