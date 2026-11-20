# Directional Shadow Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a directional light's depth map in a first pass, then sample it in the main scene
pass so opaque geometry casts hard (non-PCF) shadows onto other opaque geometry, with explicit
pass/resource dependencies under dynamic rendering.

**Architecture:** One new descriptor set (set 3: depth-comparison sampler + light-view-proj UBO) and
two pipeline variants of the existing scene shader (shadows on/off, compiled from one source via a
preprocessor macro), plus a brand-new depth-only shadow-pass pipeline with its own minimal vertex
shader. Both passes record into the SAME primary command buffer via the existing, unmodified
`command::emit_object_draws` — the shadow pass supplies a different emission closure, not a second
mesh/object resolver. `VulkanRenderer` gains `set_directional_shadow`, lazily reallocating GPU
resources only on a resolution/enable-state change while the light's view-projection matrix updates
every frame like the existing `directional_light` already does.

**Tech Stack:** Rust, `ash` 0.38 (Vulkan 1.3 core dynamic rendering), `gpu-allocator`, GLSL/`glslc`.

**Spec:** [specs/017-directional-shadow-mapping.md](../../../specs/017-directional-shadow-mapping.md)
(including its "Resolved decisions (2026-11-19)" section — read both; the plan argues from the whole
file).

## Global Constraints

- `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo build --all-targets`, and `cargo test --all-targets` must all pass clean after **every**
  commit. A shared-signature change (e.g. `GraphicsPipeline::create`'s parameters) must update every
  call site in the SAME commit as the signature change — never split across task boundaries (this
  project's `feedback_sdd_plan_must_keep_crate_compiling` lesson, hit three times during spec 015).
- Default `None`: an existing scene with no `set_directional_shadow` call keeps its current
  three-descriptor-set pipeline/shaders byte-for-byte unchanged.
- `Mipmaps`/texture handling (spec 014), instancing (spec 016), and the existing lighting model (specs
  011/013/018) are unmodified by this plan except where a shadow-enabled fragment shader multiplies
  visibility into the *directional* diffuse+specular terms only — ambient and point lights are
  unshadowed, per spec.
- Depth format candidates for the shadow map: `D32_SFLOAT` then `D16_UNORM`, requiring
  `DEPTH_STENCIL_ATTACHMENT` and `SAMPLED_IMAGE` optimal-tiling format features. No stencil formats.
- `FRAMES_IN_FLIGHT = 2` (`command.rs:20`) — one shadow depth image/view/light-VP buffer per frame
  slot, exactly as `Light`/`Uniforms`/`InstanceBuffer` already pattern.
- Set 3 layout: binding 0 combined depth-comparison sampler (`FRAGMENT`), binding 1 light-view-proj
  UBO, 64 bytes row-major (`VERTEX | FRAGMENT`). Comparison sampler: nearest min/mag, `COMPARE_OP::
  LESS_OR_EQUAL`, normalized coordinates, no anisotropy, LOD 0.
  constant/slope depth bias are `vkCmdSetDepthBias` **dynamic state** at shadow-pass record time, not
  a UBO field — the spec's "dynamic depth bias" instruction in the two-pass execution section.
- The shadow pass's own pipeline layout is `[set 3 layout]` only — no camera/texture/light sets.
- GPU-dependent tests stay `#[ignore]`d; ordinary `cargo test --all-targets` must not require a
  driver.

## Review Focus

- **A device lacking `SAMPLED_IMAGE` on both shadow-depth-format candidates, with shadows requested**:
  a reasonable caller expects `set_directional_shadow(Some(...))` to fail with a descriptive error and
  the renderer to keep working with shadows disabled (its prior pipeline/resources untouched) — not a
  panic or a silently-black shadow map. Task 3's pure format-selection tests and Task 7's
  config-rejection tests cover this.
- **A light_view_proj update on every frame (a moving/rotating light) must not reallocate the shadow
  depth image** — a reasonable caller expects `set_directional_shadow` called every frame with the
  same `resolution` to be a cheap per-frame uniform write, not a full GPU resource teardown/rebuild.
  Task 4's reallocation-decision tests cover this directly (the plan's "Resolved decisions" section
  explains why).
- **A receiver fragment outside the light's frustum (`x`/`y` outside `[-1,1]` or `z` outside `[0,1]`
  after the perspective divide)**: a reasonable caller expects that geometry to render fully lit
  (unshadowed), not black or garbage-sampled. Task 2's shader logic and Task 8's GPU test cover this
  explicitly (the spec calls this out by name).
- **Disabling shadows after they were enabled** (`Some` → `None`) must restore the exact
  pre-shadow three-set pipeline/shader variant, not merely stop sampling while still binding a
  4-set pipeline layout with set 3 unbound (invalid Vulkan usage) — Task 4's pipeline-variant-switch
  logic and Task 8's "disabled output matches the previous baseline" golden test cover this.
- **Instanced shadow casters** (spec 016's batching): a reasonable caller who already uses instancing
  for the main pass expects the shadow pass to draw the same instance runs, not one draw per object —
  Task 5 reuses `emit_object_draws`'s existing run-grouping unmodified, and Task 8's GPU test asserts
  the shadow pass's draw-call count the same way spec 016's own tests asserted the main pass's.

---

## Task 1: Generalize `PipelineDescription` for a depth-only pipeline (zero behavior change)

**Files:**
- Modify: `src/renderer/pipeline.rs:97-154` (`PipelineDescription`, `GraphicsPipeline::new`,
  `GraphicsPipeline::create`)
- Modify: `src/renderer/text.rs:166-184` (its `GraphicsPipeline::create` call site)

**Interfaces:**
- Produces (consumed by Task 4): `PipelineDescription.fragment_shader: Option<&'a [u8]>` (was
  `&'a [u8]`); `GraphicsPipeline::create`'s `color_format: vk::Format` parameter becomes
  `color_formats: &[vk::Format]` (empty slice = depth-only, no `.color_attachment_formats(...)` on
  `PipelineRenderingCreateInfo`, and the pipeline's `stages` array has exactly one entry — vertex
  only — when `fragment_shader` is `None`).

- [ ] **Step 1: Write the failing test**

```rust
// pipeline.rs
#[test]
fn shader_stage_count_is_one_without_fragment_two_with() {
    assert_eq!(shader_stage_count(None), 1);
    assert_eq!(shader_stage_count(Some(&[0u8; 4])), 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib renderer::pipeline::tests::shader_stage_count_is_one_without_fragment_two_with`
Expected: compile error — `shader_stage_count` doesn't exist yet.

- [ ] **Step 3: Implement**

Add `pub(super) fn shader_stage_count(fragment_shader: Option<&[u8]>) -> usize` (pure: `1 +
fragment_shader.is_some() as usize`) to `pipeline.rs`. Change `PipelineDescription.fragment_shader`
to `Option<&'a [u8]>`. In `GraphicsPipeline::create`, only build/read the fragment shader
module/stage when `description.fragment_shader` is `Some(_)` — the `_fragment_cleanup` guard and the
`stages` array both become conditional (use `shader_stage_count` to size the `Vec<...>`/array you
build, or push into a `Vec` instead of a fixed `[_; 2]`). Change `create`'s `color_format: vk::Format`
parameter to `color_formats: &[vk::Format]`; only call `.color_attachment_formats(color_formats)` on
`PipelineRenderingCreateInfo` when non-empty (an empty slice passed to that builder method is itself
already a no-op-equivalent to omitting it — confirm this by reading `ash`'s builder behavior, or
simply branch on `color_formats.is_empty()` if the builder call isn't a true no-op on an empty slice).
Update `GraphicsPipeline::new` (`pipeline.rs:124`) to pass `Some(FRAGMENT_SHADER)`/`&[color_format]`.
Update `text.rs:166`'s call site to pass `Some(include_bytes!(...))`/`&[color]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib renderer::pipeline`
Expected: all pipeline tests pass, including the new one.

Run: `cargo build --all-targets && cargo test --all-targets`
Expected: full crate builds; every pre-existing test still passes (both call sites updated in this
same commit — confirm neither was missed).

- [ ] **Step 5: Commit**

```bash
git add src/renderer/pipeline.rs src/renderer/text.rs
git commit -m "refactor(renderer): support an optional fragment shader and depth-only pipelines"
```

---

## Task 2: Shadow shaders — depth-only pass shader, macro-conditional main shader, compiled artifacts

**Files:**
- Create: `shaders/shadow_depth.vert` → `shaders/shadow_depth.vert.spv` (no fragment shader — a
  depth-only pipeline needs none)
- Modify: `shaders/triangle.vert` (add an `#ifdef SHADOWS_ENABLED` block)
- Modify: `shaders/triangle.frag` (add an `#ifdef SHADOWS_ENABLED` block)
- Create (compiled, not hand-written): `shaders/triangle_shadow.vert.spv`,
  `shaders/triangle_shadow.frag.spv` (same sources as `triangle.vert`/`triangle.frag`, compiled with
  `-DSHADOWS_ENABLED`)
- Modify: `shaders/triangle.vert.spv`, `shaders/triangle.frag.spv` (recompiled from the now-`#ifdef`-
  bearing sources with the macro **undefined** — must be byte-identical to before this task, since
  the macro being absent takes the exact prior code path; verify this with `git diff --stat`
  reporting no change to these two files, or investigate if it does)
- Modify: `CLAUDE.md` (document the two-variant compile step)

**Interfaces:**
- Produces (consumed by Task 4): `shaders/shadow_depth.vert.spv`,
  `shaders/triangle_shadow.{vert,frag}.spv` as `include_bytes!`-able compiled artifacts.
- Consumes: none from earlier tasks.

- [ ] **Step 1: Write the failing test**

```rust
// pipeline.rs (or a new small module if you prefer — your call) — mirrors the existing
// vertex_spirv_parses_and_starts_with_magic/fragment_spirv_parses_and_starts_with_magic tests:
#[test]
fn shadow_depth_vertex_spirv_parses_and_starts_with_magic() {
    let bytes = include_bytes!("../../shaders/shadow_depth.vert.spv");
    let words = ash::util::read_spv(&mut Cursor::new(&bytes[..])).unwrap();
    assert_eq!(words.first().copied(), Some(SPIRV_MAGIC));
}

#[test]
fn triangle_shadow_vertex_and_fragment_spirv_parse_and_start_with_magic() {
    let v = include_bytes!("../../shaders/triangle_shadow.vert.spv");
    let f = include_bytes!("../../shaders/triangle_shadow.frag.spv");
    assert_eq!(
        ash::util::read_spv(&mut Cursor::new(&v[..])).unwrap().first().copied(),
        Some(SPIRV_MAGIC)
    );
    assert_eq!(
        ash::util::read_spv(&mut Cursor::new(&f[..])).unwrap().first().copied(),
        Some(SPIRV_MAGIC)
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib renderer::pipeline`
Expected: compile error — the `.spv` files referenced by `include_bytes!` don't exist yet.

- [ ] **Step 3: Write the shaders and compile**

`shaders/shadow_depth.vert`: vertex-only, no fragment output needed (depth written automatically by
rasterization). Inputs: `layout(location = 0) in vec3 inPosition;` (binding 0, same `Vertex` layout
the main pipeline uses — reuse `Vertex::binding_description()`/its position attribute so no new CPU
vertex-input description is needed) plus the SAME four per-instance model-row attributes at locations
4-7 the main pipeline already declares (binding 1, `InstanceData`'s first four fields — reuse
`instance_binding_description()`/`instance_attribute_descriptions()`[0..4] unchanged; the 5th
`specular` attribute at location 8 is simply unused by this shader, which is fine). Uniform: `layout
(set = 0, binding = 1, row_major) uniform LightViewProj { mat4 lightViewProj; } light;` — **binding 1,
not 0**, matching the global shadow-set layout (binding 0 is the depth sampler this shader never
reads); **set 0 in this shader's own numbering**, since the shadow pipeline's pipeline layout array
has exactly one set (the shadow set) at index 0 — even though that SAME descriptor set object is
bound as "set 3" in the main pipeline's numbering. `void main()`: reconstruct `model` from the four
row vectors exactly as `triangle.vert:37` does (`transpose(mat4(...))`, same reasoning comment),
`gl_Position = light.lightViewProj * model * vec4(inPosition, 1.0);`.

`triangle.vert`'s `#ifdef SHADOWS_ENABLED` addition: add `layout(location = 5) out vec4
fragLightSpacePos;` and, inside `main()`, `#ifdef SHADOWS_ENABLED` / `fragLightSpacePos =
light.lightViewProj * worldPos;` / `#endif` (the shadow set's own `layout(set = 3, binding = 1,
row_major) uniform LightViewProj { mat4 lightViewProj; } light;` declaration also goes inside the
same `#ifdef`, since it only exists in the enabled variant).

`triangle.frag`'s `#ifdef SHADOWS_ENABLED` addition: `layout(location = 5) in vec4 fragLightSpacePos;`
input (matching the vertex shader's output location), the set-3 declarations `layout(set = 3, binding
= 0) uniform sampler2DShadow shadowMap;` and the light-view-proj UBO mirror (only `lightViewProj`'s
bias fields if you choose to also pass bias via this UBO — **no**, bias is dynamic state per the
Global Constraints, not a shader input at all), all inside `#ifdef`. Add a `float shadowVisibility
(vec4 lightSpacePos)` function (also inside the `#ifdef`): divide `xyz` by `w` only when `w` is
positive and non-degenerate (else return `1.0`, fully lit — per spec, a degenerate projection must not
produce a black/garbage shadow); map `xy` from `[-1,1]` to `[0,1]` (do **not** flip Y a second time —
the spec's explicit warning); if the mapped `xy` or the resulting `z` falls outside `[0,1]`, return
`1.0` (fully lit, per spec's "explicitly fully lit" rule for out-of-frustum receivers); otherwise
`texture(shadowMap, vec3(uv, z))` (the hardware depth-compare sampler's built-in `LESS_OR_EQUAL`
comparison against the stored depth does the hard-shadow test, returning 0 or 1 already — no manual
`<=` needed). In `main()`, `#ifdef SHADOWS_ENABLED` wraps a call to `shadowVisibility(fragLightSpacePos)`
whose result multiplies ONLY the directional diffusion contribution (`light.color * diff * albedo.rgb`)
and the directional specular `highlight` term — leave `light.ambient * albedo.rgb` and the point-light
loop's contributions untouched, per spec. `#else` (or simply: outside any `#ifdef`, visibility = 1.0
when the macro is undefined) leaves today's `shaded` computation bit-for-bit identical.

Compile all four artifacts:
```bash
glslc shaders/shadow_depth.vert -o shaders/shadow_depth.vert.spv
glslc shaders/triangle.vert -o shaders/triangle.vert.spv
glslc shaders/triangle.frag -o shaders/triangle.frag.spv
glslc -DSHADOWS_ENABLED shaders/triangle.vert -o shaders/triangle_shadow.vert.spv
glslc -DSHADOWS_ENABLED shaders/triangle.frag -o shaders/triangle_shadow.frag.spv
```
Add the two-variant compile step to `CLAUDE.md`'s existing shader-recompile note (`pipeline.rs:15`'s
comment references it too — update that comment's cross-reference if it goes stale).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib renderer::pipeline`
Expected: all SPIR-V magic-number tests pass (existing two plus the three new ones).

Run: `git diff --stat shaders/triangle.vert.spv shaders/triangle.frag.spv`
Expected: **no output** (byte-identical) — if these two files changed, the `#ifdef` additions leaked
into the undefined-macro code path; find and fix the leak before committing.

- [ ] **Step 5: Commit**

```bash
git add shaders/ CLAUDE.md src/renderer/pipeline.rs
git commit -m "feat(shaders): add depth-only shadow-pass shader and shadow-enabled scene variant"
```

---

## Task 3: `shadow.rs` — `DirectionalShadow`, validation, and per-frame GPU shadow resources

**Files:**
- Create: `src/renderer/shadow.rs`
- Modify: `src/renderer/mod.rs` (module declaration `pub mod shadow;`, near the other `mod`
  declarations at the top)
- Modify: `src/lib.rs` (re-export `DirectionalShadow`)

**Interfaces:**
- Consumes: `crate::renderer::buffer::Buffer::new_host_visible_mapped` (existing, from `buffer.rs`),
  same shape `per_frame_uniform.rs`/`image.rs` already use for host-visible mapped buffers.
- Produces (consumed by Tasks 4-6):
  - `pub struct DirectionalShadow { pub light_view_proj: Mat4, pub resolution: u32, pub
    constant_bias: f32, pub slope_bias: f32 }` (exact shape from the spec's Public API section),
    re-exported from `lib.rs`.
  - `pub(crate) fn validate_directional_shadow(shadow: &DirectionalShadow) -> Result<()>` — rejects
    `resolution == 0`; rejects any non-finite element of `light_view_proj` (all 16 components, via
    `m2s2-math`'s `Index<usize>` row access, same pattern `light.rs`'s direction-validation uses);
    rejects non-finite `constant_bias`/`slope_bias` (both may be any other finite value, signed).
  - `pub(crate) fn select_shadow_depth_format(candidates: &[(vk::Format,
    vk::FormatFeatureFlags)]) -> Option<vk::Format>` — same shape as `depth.rs`'s
    `select_depth_format`, but requires BOTH `DEPTH_STENCIL_ATTACHMENT` and `SAMPLED_IMAGE` (not
    `SAMPLED_IMAGE_FILTER_LINEAR` — the comparison sampler uses `NEAREST`, no linear filtering
    requirement).
  - `pub(crate) const SHADOW_DEPTH_FORMAT_CANDIDATES: [vk::Format; 2] = [vk::Format::D32_SFLOAT,
    vk::Format::D16_UNORM];`
  - `pub(crate) struct ShadowResources` — owns, per frame slot (`[_; FRAMES_IN_FLIGHT]`): a depth
    image + view + allocation at `resolution × resolution`, a light-view-proj host-visible mapped
    `Buffer` (64 bytes), a descriptor set (set-3 shape: binding 0 = that slot's image view + the one
    shared comparison sampler, binding 1 = that slot's light-VP buffer); one shared descriptor-set
    layout (2 bindings, as specified), one shared descriptor pool (sized for `FRAMES_IN_FLIGHT` sets
    of each type), one shared `vk::Sampler` (nearest min/mag, `COMPARE_OP::LESS_OR_EQUAL`,
    `compare_enable(true)`, `CLAMP_TO_EDGE` addressing, no anisotropy, `min_lod`/`max_lod` 0), and a
    `[bool; FRAMES_IN_FLIGHT]` "has this slot ever been rendered into" tracker (all `false` at
    construction).
    - `ShadowResources::new(device: &VulkanDevice, instance: &VulkanInstance, allocator: &mut
      Allocator, resolution: u32) -> Result<Self>` — queries
      `get_physical_device_format_properties` for each candidate, calls
      `select_shadow_depth_format`, fails with a descriptive `VulkanError::InitializationError`
      naming the missing capability if neither candidate qualifies; rolls back every
      partially-constructed resource on any later failure (mirrors `StoredTexture::new`'s existing
      rollback shape).
    - `descriptor_set_layout(&self) -> vk::DescriptorSetLayout`
    - `descriptor_set(&self, frame: usize) -> vk::DescriptorSet`
    - `image(&self, frame: usize) -> vk::Image`, `view(&self, frame: usize) -> vk::ImageView`,
      `format(&self) -> vk::Format`
    - `update(&mut self, frame: usize, light_view_proj: Mat4) -> Result<()>` — writes that frame's
      light-VP buffer only (bias is NOT stored here — it's dynamic state at record time, per Global
      Constraints).
    - `mark_used(&mut self, frame: usize)` / `was_used(&self, frame: usize) -> bool` — the
      first-use/reuse tracker Task 5's barrier logic reads and updates.
    - `destroy(self, device: &VulkanDevice, allocator: &mut Allocator)`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn validate_directional_shadow_rejects_zero_resolution() {
    let mut s = valid_shadow();
    s.resolution = 0;
    assert!(validate_directional_shadow(&s).is_err());
}

#[test]
fn validate_directional_shadow_rejects_non_finite_matrix_or_bias() {
    let base = valid_shadow();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut s = base;
        s.light_view_proj[0][0] = bad; // any one element suffices
        assert!(validate_directional_shadow(&s).is_err());
        let mut s = base;
        s.constant_bias = bad;
        assert!(validate_directional_shadow(&s).is_err());
        let mut s = base;
        s.slope_bias = bad;
        assert!(validate_directional_shadow(&s).is_err());
    }
}

#[test]
fn validate_directional_shadow_accepts_signed_finite_bias() {
    let mut s = valid_shadow();
    s.constant_bias = -1.5;
    s.slope_bias = -0.5;
    assert!(validate_directional_shadow(&s).is_ok());
}

#[test]
fn select_shadow_depth_format_prefers_d32_requires_both_features() {
    let both = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE;
    let candidates = [(vk::Format::D32_SFLOAT, both), (vk::Format::D16_UNORM, both)];
    assert_eq!(select_shadow_depth_format(&candidates), Some(vk::Format::D32_SFLOAT));
}

#[test]
fn select_shadow_depth_format_falls_back_when_first_lacks_sampled_image() {
    let both = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE;
    let attachment_only = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
    let candidates = [(vk::Format::D32_SFLOAT, attachment_only), (vk::Format::D16_UNORM, both)];
    assert_eq!(select_shadow_depth_format(&candidates), Some(vk::Format::D16_UNORM));
}

#[test]
fn select_shadow_depth_format_returns_none_when_no_candidate_qualifies() {
    let candidates = [
        (vk::Format::D32_SFLOAT, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
        (vk::Format::D16_UNORM, vk::FormatFeatureFlags::SAMPLED_IMAGE),
    ];
    assert_eq!(select_shadow_depth_format(&candidates), None);
}
```

(`fn valid_shadow() -> DirectionalShadow` is a small test helper returning `Mat4::IDENTITY`,
`resolution: 1024`, small positive biases — your call on exact values.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::shadow`
Expected: compile error — module/functions don't exist yet.

- [ ] **Step 3: Implement**

Write `shadow.rs` as specified. For the resource-construction/rollback shape, follow
`texture.rs`'s `StoredTexture::new` and `depth.rs`'s `DepthResources::new` closely — both already in
this codebase and already reviewed for spec 012's ownership contract. `ShadowResources::new` needs
`&VulkanInstance` only for the format-property query (mirrors `depth.rs:31-47`); the descriptor
set/pool/sampler/buffers use `&VulkanDevice`/`&mut Allocator` exactly like `per_frame_uniform.rs`'s
patterns, just with two binding types in one layout instead of one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib renderer::shadow`
Expected: all 6 new tests pass.

Run: `cargo build --all-targets && cargo test --all-targets`
Expected: full crate builds; no pre-existing test regresses.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/shadow.rs src/renderer/mod.rs src/lib.rs
git commit -m "feat(renderer): directional shadow configuration and per-frame GPU resources"
```

---

## Task 4: `VulkanRenderer` wiring — `set_directional_shadow`, lazy reallocation, pipeline variant switch

**Files:**
- Modify: `src/renderer/mod.rs` (new fields, `set_directional_shadow`, pending-config application
  inside `render_objects_with_text`, pipeline-variant rebuild)
- Modify: `src/renderer/pipeline.rs` (a `GraphicsPipeline::new_shadow_depth` constructor for the
  depth-only shadow pipeline, and generalizing `GraphicsPipeline::new`/wherever the main pipeline is
  built to select the shadow-enabled vs. disabled shader/set-layout variant)

**Interfaces:**
- Consumes: `shadow::{DirectionalShadow, validate_directional_shadow, ShadowResources,
  SHADOW_DEPTH_FORMAT_CANDIDATES, select_shadow_depth_format}` (Task 3);
  `pipeline::PipelineDescription`'s now-`Option` fragment shader and slice color formats (Task 1);
  the compiled shader bytes from Task 2 (`include_bytes!("../../shaders/shadow_depth.vert.spv")`,
  etc.).
- Produces (consumed by Task 5): `VulkanRenderer` fields `shadow: Option<shadow::ShadowResources>`,
  `shadow_pipeline: Option<pipeline::GraphicsPipeline>` (the depth-only pass pipeline),
  `pending_shadow_config: Option<Option<u32>>` (resolution-only; `None` = nothing pending, `Some
  (None)` = disable, `Some(Some(resolution))` = (re)allocate at `resolution`),
  `current_shadow_values: Option<(Mat4, f32, f32)>` (light_view_proj, constant_bias, slope_bias —
  always present when shadows are enabled, uploaded every frame), and the main `pipeline` field now
  possibly built from the shadow-enabled shader variant + 4 set layouts.

- [ ] **Step 1: Write the failing tests**

```rust
// mod.rs — pure reallocation-decision logic (no GPU needed):
#[test]
fn shadow_reallocation_needed_on_first_enable_and_resolution_change_only() {
    assert!(shadow_reallocation_needed(None, Some(1024)));       // first enable
    assert!(shadow_reallocation_needed(Some(1024), Some(2048))); // resolution changed
    assert!(shadow_reallocation_needed(Some(1024), None));       // disabling
    assert!(!shadow_reallocation_needed(Some(1024), Some(1024))); // same resolution: no-op
    assert!(!shadow_reallocation_needed(None, None));            // staying disabled: no-op
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib renderer::tests::shadow_reallocation_needed_on_first_enable_and_resolution_change_only`
Expected: compile error — `shadow_reallocation_needed` doesn't exist yet.

- [ ] **Step 3: Implement**

Add `fn shadow_reallocation_needed(current_resolution: Option<u32>, requested_resolution: Option<u32>)
-> bool` (pure: `current_resolution != requested_resolution`) to `mod.rs`, alongside the existing
`validate_object_count`.

`set_directional_shadow(&mut self, shadow: Option<DirectionalShadow>) -> Result<()>`: `None` →
compute `requested_resolution = None`; `Some(s)` → `shadow::validate_directional_shadow(&s)?`, then
`requested_resolution = Some(s.resolution)`. Compare against the resolution implied by
`self.pending_shadow_config` if set, else by `self.shadow`'s live resolution if any, else `None`. If
`shadow_reallocation_needed(...)`, set `self.pending_shadow_config = Some(requested_resolution.map
(|r| /* carries whatever shape you chose above */))`. **Always**, regardless of reallocation,
update `self.current_shadow_values = shadow.map(|s| (s.light_view_proj, s.constant_bias,
s.slope_bias))` when `shadow.is_some()` (leave it as-is, or clear it, when disabling — your call,
but document which; clearing is simplest since a disabled pass never reads it). On validation
failure, change nothing (same "reject without touching prior state" contract `set_directional_light`/
`set_point_lights` already honor).

Apply `pending_shadow_config` at the same point in `render_objects_with_text` that `pending_resize`
is applied (`mod.rs:418-469`) — same ordering (after up-front object/camera validation, before
acquiring a frame), same "keep old resources valid on allocation failure" contract: build the
replacement `ShadowResources`/`shadow_pipeline`/rebuilt main `pipeline` into local variables first,
and only swap them into `self` once every fallible step succeeded; on `None` (disabling), destroy the
live `self.shadow`/`self.shadow_pipeline` and rebuild `self.pipeline` back to the 3-set/disabled
shader variant. Add `GraphicsPipeline::new_shadow_depth(device: &VulkanDevice, depth_format:
vk::Format, set_layouts: &[vk::DescriptorSetLayout]) -> Result<Self>` to `pipeline.rs`, a thin
wrapper around `create` passing `color_formats: &[]`, `fragment_shader: None`, the
`shadow_depth.vert.spv` bytes, the SAME `Vertex`+instance bindings/attributes the main pipeline uses
(reuse the existing helper functions — no new vertex-input description needed), `depth_test: true`,
`cull_mode: vk::CullModeFlags::BACK` (per spec's "back-face culling initially"), `blend: false`,
no push ranges. Generalize wherever the main `pipeline` is constructed (both in `init` and in the
resize path) to pick `triangle.{vert,frag}.spv`/3 set layouts when `self.shadow.is_none()` after
this apply step, or `triangle_shadow.{vert,frag}.spv`/4 set layouts (camera, texture, light, shadow)
when `self.shadow.is_some()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib renderer`
Expected: new test passes; full lib test suite still green.

Run: `cargo build --all-targets && cargo test --all-targets`
Expected: full crate builds (this task's real GPU-resource-construction code paths aren't exercised
by CPU tests — that's Task 8 — but the crate must compile and every existing test must still pass).

- [ ] **Step 5: Commit**

```bash
git add src/renderer/mod.rs src/renderer/pipeline.rs
git commit -m "feat(renderer): set_directional_shadow with lazy GPU resource reallocation"
```

---

## Task 5: Two-pass recording — `record_shadow_pass` shared helper, wired into `FrameContext`

**Files:**
- Modify: `src/renderer/command.rs` (new `record_shadow_pass` function; `FrameContext::record`/
  `draw_frame` call it before the existing scene recording when shadows are enabled)

**Interfaces:**
- Consumes: `shadow::ShadowResources` (Task 3: `image`/`view`/`format`/`descriptor_set`/
  `was_used`/`mark_used`), `VulkanRenderer`'s `shadow`/`shadow_pipeline`/`current_shadow_values`
  fields (Task 4), the UNCHANGED `command::emit_object_draws`/`ObjectCommand` (already exists).
- Produces (consumed by Task 6): `pub(super) fn record_shadow_pass(dev: &ash::Device, cmd:
  vk::CommandBuffer, shadow_pipeline: &GraphicsPipeline, shadow: &ShadowResources, frame: usize,
  objects: &[Object], owner: u64, meshes: &[StoredMesh], instance_buffer: vk::Buffer,
  constant_bias: f32, slope_bias: f32) -> ()` — a free function (not a `FrameContext` method, so
  `headless.rs`'s separate recording path can call it too in Task 6), parameterized purely over raw
  handles the same way `record`'s existing barrier helpers (`depth_stages()`, `color_subresource_
  range()`) already are.

- [ ] **Step 1: Write the failing tests**

```rust
// command.rs — pure barrier-scope selection, no GPU needed (mirrors the existing
// depth_stages()/acquire_wait_orders_the_color_layout_transition precedent):
#[test]
fn shadow_barrier_old_layout_is_undefined_on_first_use_read_only_on_reuse() {
    assert_eq!(shadow_attachment_old_layout(false), vk::ImageLayout::UNDEFINED);
    assert_eq!(
        shadow_attachment_old_layout(true),
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
    );
}

#[test]
fn shadow_barrier_src_access_is_empty_on_first_use_shader_read_on_reuse() {
    assert_eq!(shadow_attachment_src_access(false), vk::AccessFlags::empty());
    assert_eq!(shadow_attachment_src_access(true), vk::AccessFlags::SHADER_READ);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::command::tests::shadow_barrier`
Expected: compile error — functions don't exist yet.

- [ ] **Step 3: Implement**

Add the two pure helpers `fn shadow_attachment_old_layout(was_used: bool) -> vk::ImageLayout` /
`fn shadow_attachment_src_access(was_used: bool) -> vk::AccessFlags` exactly as the tests specify
(mirrors the spec's barrier-scope section 1: "Transition its depth image from UNDEFINED on first
use, or DEPTH_STENCIL_READ_ONLY_OPTIMAL on reuse... order prior fragment sampling before attachment
read/write at EARLY | LATE_FRAGMENT_TESTS").

Implement `record_shadow_pass` following the spec's explicit 1-3 numbered sequence (its "Explicit
two-pass execution" section, steps 1-3 — step 4 is the EXISTING main-pass recording, unchanged
except for the two additions Task 4 already wired: binding set 3 and using the shadow-enabled
pipeline):
1. Barrier using `shadow_attachment_old_layout(shadow.was_used(frame))` →
   `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, source stage `EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS` on
   reuse (`FRAGMENT_SHADER` is also a valid source stage choice for the reuse case per the spec's
   "order prior fragment sampling" wording — pick whichever matches the main pass's actual shadow-
   sampling stage, which is `FRAGMENT_SHADER`) or `TOP_OF_PIPE` on first use, destination stage
   `EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS`, destination access
   `DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE`.
2. `cmd_begin_rendering` with a depth-only `RenderingInfo` (no color attachments) at `shadow.
   resolution × resolution`, depth attachment `load_op: CLEAR` to `1.0`, `store_op: STORE` (must be
   sampled afterward). `cmd_bind_pipeline` (shadow_pipeline), `cmd_bind_descriptor_sets` (set 0 in
   this pipeline's own numbering = the shadow set, `shadow.descriptor_set(frame)`),
   `cmd_set_viewport`/`cmd_set_scissor` to the shadow resolution, `cmd_set_depth_bias(cmd,
   constant_bias, 0.0, slope_bias)` (the `clamp` parameter: `0.0`, per the spec not mentioning a
   clamp value), then call `emit_object_draws(objects, owner, meshes, &[], white_set_placeholder,
   instance_buffer, |command| match command { BindInstanceBuffer(buf) => bind to binding 1 (same as
   the main pass) ; BindMesh(vb, ib) => bind vertex/index buffers (same) ; BindTexture(_) => {} (no-
   op — nothing to bind) ; DrawIndexed { index_count, instance_count, first_instance } =>
   cmd_draw_indexed(..., 0, 0, first_instance) })` — note `emit_object_draws` requires a
   `texture_sets`/`white_set` argument even though this pass ignores the resolved value entirely; pass
   `&[]`/any placeholder `vk::DescriptorSet` since `BindTexture`'s payload is simply dropped in this
   closure (every object's texture handle was already validated once, up-front, by the caller before
   either pass records — re-resolving it here costs nothing and keeps one resolver, per the Review
   Focus item).
3. `cmd_end_rendering`, then barrier `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` →
   `DEPTH_STENCIL_READ_ONLY_OPTIMAL`, source stage `EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS`,
   source access `DEPTH_STENCIL_ATTACHMENT_WRITE`, destination stage `FRAGMENT_SHADER`, destination
   access `SHADER_READ`. Call `shadow.mark_used(frame)` after recording (or have the caller do it —
   your call on which side owns the mutation, document it either way).

Wire this into `FrameContext::record`/`draw_frame`: when `shadow`/`shadow_pipeline` are `Some(_)`,
call `record_shadow_pass` with `frame`'s index BEFORE the existing scene-pass barriers/begin-
rendering, and — per the spec's step 4 — explicitly re-issue `cmd_set_viewport`/`cmd_set_scissor` at
the MAIN extent and re-bind the main scene pipeline right after, since the shadow pass just changed
both. Bind set 3 (`shadow.descriptor_set(frame)`) alongside sets 0/2 in the existing "bind frame-
global descriptor sets" block (`command.rs:421-436`), only when shadows are enabled. Upload
`current_shadow_values`'s `light_view_proj` into `shadow.update(frame, ...)` in the same per-frame
update block that already writes `uniforms`/`light`/`instance_buffer` (`command.rs:205-226`), only
when shadows are enabled — this is `draw_frame`'s responsibility, called from `mod.rs`'s
`render_objects_with_text` passing `self.shadow.as_mut()`/`self.current_shadow_values`. Exclude
shadow-pass draws from `scene_draw_calls`/`submitted_indices`/`scene_instances` — these counters
already come from a SEPARATE re-walk of `emit_object_draws` in `mod.rs:511-532` that only mirrors the
MAIN pass's recording, so no change is needed there as long as `record_shadow_pass`'s own
`emit_object_draws` call doesn't feed into that counting walk (it doesn't — they're entirely separate
call sites).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib renderer::command`
Expected: new barrier-scope tests pass; all pre-existing command.rs tests still pass.

Run: `cargo build --all-targets && cargo test --all-targets`
Expected: full crate builds; no regression.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/command.rs
git commit -m "feat(renderer): record the shadow depth pass before the main scene pass"
```

---

## Task 6: `HeadlessRenderer` mirroring

**Files:**
- Modify: `src/renderer/headless.rs` (mirror Task 3/4/5's fields and wiring: `shadow`,
  `shadow_pipeline`, `pending_shadow_config`, `current_shadow_values`, a `pub(crate) fn
  set_directional_shadow`, and a call to `command::record_shadow_pass` from its own recording
  function)

**Interfaces:**
- Consumes: everything Tasks 3-5 produced, applied to `HeadlessRenderer` the same way `VulkanRenderer`
  already uses it (this task is the mechanical mirror once the pattern exists — `HeadlessRenderer`'s
  constructor/field list already mirrors `VulkanRenderer`'s for every other renderer-owned resource,
  per `headless.rs`'s established convention).
- Produces (consumed by Task 8): `HeadlessRenderer::set_directional_shadow`, so GPU tests can
  exercise shadows in the headless/golden-image path.

- [ ] **Step 1: Write the failing test**

```rust
// headless.rs, #[cfg(test)] mod tests, #[ignore]:
#[test]
fn set_directional_shadow_then_disable_round_trips_without_terminal_state() {
    // Construct a HeadlessRenderer, call set_directional_shadow(Some(valid_shadow())), render a
    // frame successfully, call set_directional_shadow(None), render again successfully. Assert
    // renderer isn't terminal and validation_error_count() == 0 throughout.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib renderer::headless -- --ignored set_directional_shadow_then_disable_round_trips_without_terminal_state`
Expected: compile error — `set_directional_shadow` doesn't exist on `HeadlessRenderer` yet.

- [ ] **Step 3: Implement**

Mirror `VulkanRenderer`'s Task 4/5 wiring onto `HeadlessRenderer`: same field additions, same
`set_directional_shadow` body (the instance/device/allocator access pattern is identical — both
renderer types already hold these as sibling fields, confirmed during spec 014's grounding), same
call to `command::record_shadow_pass` inserted into `HeadlessRenderer`'s own recording function at
the equivalent point Task 5 inserted it into `FrameContext::record`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib renderer::headless -- --ignored set_directional_shadow_then_disable_round_trips_without_terminal_state`
Expected: PASS (on a machine with a driver) or correctly skipped (`cargo test --all-targets` without
`--ignored`).

Run: `cargo build --all-targets && cargo test --all-targets`
Expected: full crate builds; no regression.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/headless.rs
git commit -m "feat(renderer): mirror directional shadow support onto HeadlessRenderer"
```

---

## Task 7: CPU failure-path and ownership-contract tests

**Files:**
- Modify: `src/renderer/shadow.rs` (tests only, or a small addendum fix if a real gap surfaces —
  same escape valve spec 014's equivalent task used)
- Modify: `src/renderer/headless.rs` (tests only, `#[cfg(test)]`, `#[ignore]`)

**Interfaces:**
- Consumes: `set_directional_shadow`, `ShadowResources` (Tasks 3-6).
- Produces: nothing new; pure test coverage proving the ownership/rejection contract Task 4's
  reuse of the existing `pending_resize`-style rollback shape already establishes.

- [ ] **Step 1: Write the failing tests**

```rust
// headless.rs, #[ignore]:
#[test]
fn set_directional_shadow_rejects_zero_resolution_without_changing_prior_config() {
    // Enable shadows successfully at resolution 1024, then attempt resolution 0 — must error,
    // and a subsequent render must still use the prior 1024 configuration (not disabled, not
    // reallocated), proving set_directional_shadow's reject-without-mutating contract.
}

#[test]
fn shadow_reallocation_preserves_renderer_on_allocation_failure() {
    // Exercise the Task 4 rollback path: if a real injectable-failure fixture exists elsewhere in
    // this codebase for allocation failure (check mesh.rs/texture.rs's existing failure-injection
    // tests for the pattern this project already uses), reuse it; otherwise this test may need to
    // be a design-level assertion that the code path keeps `self.shadow`/`self.pipeline` untouched
    // until every fallible step of reallocation succeeds (read Task 4's own implementation to
    // confirm by inspection if no injectable failure fixture is available — note which in your
    // report rather than skipping the requirement silently).
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::headless -- --ignored set_directional_shadow_rejects`
Expected: fails for the right reason (feature missing or assertion not yet true), not a typo.

- [ ] **Step 3: Implement** (tests only; see note above about addendum fixes if a gap surfaces)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --all-targets` (CPU-only subset) and, on a machine with a driver,
`cargo test --lib renderer::headless -- --ignored`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/shadow.rs src/renderer/headless.rs
git commit -m "test(renderer): cover shadow configuration rejection and reallocation ownership"
```

---

## Task 8: GPU acceptance — golden-image tests and cube example demo

**Files:**
- Modify: `src/renderer/headless.rs` (GPU golden-image tests, `#[ignore]`)
- Create: `testdata/golden/shadow_*.png` (generated by the user on real hardware, not committed by
  the implementer — see Step 5)
- Modify: `examples/cube.rs` (add a ground-plane receiver and enable shadows on the existing cube(s))

**Interfaces:**
- Consumes: `HeadlessRenderer::set_directional_shadow` (Task 6), `golden::assert_matches_golden`
  (existing infra, tolerance literal `2` per convention), `RendererStats.scene_draw_calls` (existing,
  for the instanced-casters draw-count assertion).
- Produces: nothing consumed by a later task (final task).

- [ ] **Step 1: Write the failing tests**

```rust
// headless.rs, #[ignore]:
#[test]
fn shadow_disabled_output_matches_pre_shadow_baseline() {
    // A cube + receiver plane scene with shadows never enabled must match an existing (or newly
    // committed) non-shadow golden exactly — proving the disabled path is byte-for-byte the old
    // pipeline/shader, not merely "visually similar." assert_eq!(renderer.validation_error_count(), 0).
}

#[test]
fn shadow_enabled_cube_casts_onto_plane_matches_golden() {
    // Same scene, shadows enabled: assert_matches_golden against shadow_cube_on_plane.png.
    // assert_eq!(renderer.validation_error_count(), 0) — required by the spec's "no validation or
    // synchronization errors with two frames in flight" GPU test-plan line; add this assertion to
    // every new golden test in this task, not just this one.
}

#[test]
fn shadow_depth_contents_show_a_near_caster_and_a_far_cleared_background() {
    // Per the spec: "inspect real shadow depth contents... a lit screenshot alone does not prove
    // the shadow pass works." Add a small test-only readback helper (mirrors spec 014's
    // mip_readback.rs shape: barrier the shadow depth image at frame N from
    // DEPTH_STENCIL_READ_ONLY_OPTIMAL -> TRANSFER_SRC_OPTIMAL, cmd_copy_image_to_buffer into a
    // host-visible staging buffer, barrier back, read the mapped bytes) exposed as a
    // `pub(crate) fn shadow_image(&self, frame: usize) -> vk::Image` accessor on HeadlessRenderer
    // (or reuse ShadowResources::image directly if already pub(crate) — check Task 3's shape) plus
    // a `read_back_depth_texel(device, allocator, image, format, x, y, resolution) -> Result<f32>`
    // helper. Render a scene with one caster directly between the light and a receiver plane;
    // assert the texel under the caster reads a depth value closer to 0.0 (near the light) than a
    // texel in an empty region of the depth map, which should read close to 1.0 (the clear value,
    // confirming nothing wrote there). This is the one test in this task that inspects the shadow
    // MAP's own contents rather than the final shaded color output.
}

#[test]
fn out_of_frustum_receiver_is_fully_lit_not_black() {
    // A receiver positioned outside the light's view-projection frustum must render identically to
    // the shadow-disabled case for that fragment region — proving the "explicitly fully lit" rule.
}

#[test]
fn instanced_shadow_casters_match_main_pass_draw_call_reduction() {
    // Reuse spec 016's instancing scenario with shadows enabled; assert the SHADOW pass also
    // produces the reduced draw-call count (not one draw per instance) — read-back via whatever
    // counting mechanism you add (the plan doesn't thread shadow-pass draw calls through
    // RendererStats; add a minimal test-only counter the same way scene_draw_calls was added in a
    // prior spec, or inspect via a different mechanism — your call, document which).
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib renderer::headless -- --ignored shadow`
Expected: compile errors or missing-golden errors — both expected before implementation/goldens exist.

- [ ] **Step 3: Implement**

Write the four tests above (plus any others you judge necessary from the spec's GPU test-plan list:
moving light, resize, resolution change, bias tradeoffs, text alongside shadows, teardown — use
judgment on how many distinct golden images are worth committing vs. which scenarios are adequately
covered by a shared one). Update `examples/cube.rs`: add a simple quad "ground plane" mesh the
existing cube(s) can cast onto, call `set_directional_shadow` with a reasonable resolution (1024,
per the spec's proposed example value) and small positive bias defaults (document your chosen
starting values — e.g. `constant_bias: 1.25, slope_bias: 1.75` are common Vulkan starting points;
these are explicitly expected to need hardware-verified tuning, per the spec's "bias defaults...
recorded after visual testing").

- [ ] **Step 4: Run test to verify it passes**

CPU-only (no driver): `cargo build --all-targets && cargo test --all-targets` — confirms compilation
and that ignored tests are skipped cleanly; `cargo build --examples` picks up the cube changes.

GPU (hands off to the user, same pattern as every prior GPU-tested spec this project has shipped): on
a machine with a Vulkan driver, run `M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless --
--ignored shadow` once to create the new goldens, re-run the same tests without the env var to
confirm they pass against what was just written, then run the FULL `cargo test --lib renderer::
headless -- --ignored` (every pre-existing ignored test, not just this task's) and confirm every
pre-existing golden PNG is still byte-identical via `git diff --quiet testdata/golden/<file>` for
each file already in the repo — the project's established zero-regression check.

- [ ] **Step 5: Commit**

Two commits, matching the project's established code/goldens split:

```bash
git add src/renderer/headless.rs examples/cube.rs
git commit -m "test(renderer): GPU shadow-pass golden-image acceptance tests and cube demo"
# After the user generates goldens on real hardware and the zero-regression check above passes:
git add testdata/golden/shadow_*.png
git commit -m "test(golden): add directional shadow acceptance images"
```
