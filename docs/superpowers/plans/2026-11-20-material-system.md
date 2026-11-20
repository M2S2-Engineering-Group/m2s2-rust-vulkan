# Material system Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `Material` type (texture + base-color tint + specular) that replaces `Object`'s separate `texture`/`specular` fields, thread a new per-instance `base_color` through the GPU pipeline, and have `src/model.rs`'s glTF loader surface `baseColorFactor` into it — closing the gap where an untextured, flat-tinted glTF material renders plain white today.

**Architecture:** One new pure-Rust module (`src/renderer/material.rs`) defines `Material`/`validate_material`, independent of everything else (Task 1). A second, larger task migrates `Object`, `InstanceData`, both shader variants, and every call site in one atomic change, since the crate won't compile with `Object.texture` removed until every consumer is updated together (Task 2 — this is a single interlocked signature change, not several that must each compile independently). Remaining tasks extend the glTF loader (Task 3), wire the real asset's factor into the example (Task 4), add GPU acceptance goldens (Task 5), and close out the spec (Task 6).

**Tech Stack:** Rust, `ash`/Vulkan (existing pipeline/instance-buffer machinery), GLSL/`glslc` (shader recompilation), existing `m2s2-math` types.

**Spec:** [specs/021-material-system.md](../../../specs/021-material-system.md)

## Global Constraints

- `Material::default()` must produce byte-identical `InstanceData` to today's pre-spec `Object { texture: None, specular: SpecularMaterial::default() }`: `texture: None`, `base_color: Vec3::new(1.0, 1.0, 1.0)`, `specular: SpecularMaterial::default()`.
- `base_color` components must be finite and `>= 0.0` (same rule `validate_specular` already applies to specular color).
- No shading-math change beyond one multiply in the fragment shader. No metallic/roughness, emissive, normal/occlusion maps, alpha blending/cutout, or double-sided culling — all out of scope (separate future "PBR materials" spec).
- `base_color_factor` on `LoadedMesh` is always populated: glTF's own default `[1,1,1,1]` when a material/document doesn't set it, and `(1,1,1)` when a primitive has no material at all (spec 020's existing `texture: None` case). Alpha component is dropped.
- Shader location renumbering: `fragBaseColor` is a new unconditional varying at location 5; `SHADOWS_ENABLED`'s `fragLightSpacePos` moves from location 5 to location 6. All four `.spv` outputs (`triangle.vert/frag`, `triangle_shadow.vert/frag`) must be recompiled; `shadow_depth.vert`/`.spv` needs no change (already omits unused attributes — see spec's Vulkan-objects section).
- Zero regression: every pre-existing golden PNG under `testdata/golden/` must remain byte-identical after this spec's changes.

## Review Focus

- An untextured mesh whose glTF material sets a non-default `baseColorFactor` with no `baseColorTexture` at all — the headline scenario this spec exists to fix; a loader that only handles "factor + texture together" would miss exactly this case.
- Two objects sharing the same mesh+texture (so they still batch into one instanced draw per spec 016) but with *different* `base_color` tints — each instance's `InstanceData` must carry its own distinct tint; a batching implementation that assumes uniform material across a batch would silently apply only the first object's tint to the whole run.
- A tint with a very small (near-zero, not pure-zero) component on an otherwise normally-lit textured surface — a reasonable person expects the surface to visibly darken toward that channel, not error, clip unexpectedly, or desaturate in an unrelated way.
- Forgetting to recompile one of the four `.spv` outputs after the shader-location renumbering — produces a pipeline/attribute mismatch that a quick `cargo build` won't catch (SPIR-V is precompiled and committed, not built from source), only a real pipeline-creation or validation-layer failure at runtime.
- Validation ordering: `validate_material` must run in the same up-front, whole-frame-rejecting loop `validate_specular` runs in today (reject the entire frame before touching the GPU), not scattered into a later, partially-applied check.

---

## Task 1: `Material` type and validation

**Files:**
- Create: `src/renderer/material.rs`
- Modify: `src/renderer/mod.rs` (add `mod material;`)
- Test: inline `#[cfg(test)] mod tests` in `src/renderer/material.rs`

**Interfaces:**
- Produces (used by Task 2): `pub struct Material { pub texture: Option<TextureId>, pub base_color: Vec3, pub specular: SpecularMaterial }`, `impl Default for Material`, `pub(crate) fn validate_material(material: &Material) -> Result<()>`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn default_matches_pre_material_object_defaults() {
    let m = Material::default();
    assert_eq!(m.texture, None);
    assert_eq!(m.base_color, Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(m.specular, SpecularMaterial::default());
}

#[test]
fn validate_material_rejects_non_finite_or_negative_base_color() {
    let base = Material::default();
    assert!(validate_material(&base).is_ok());
    for bad in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for i in 0..3 {
            let mut m = base;
            m.base_color[i] = bad;
            assert!(validate_material(&m).is_err());
        }
    }
}

#[test]
fn validate_material_delegates_specular_validation() {
    let mut m = Material::default();
    m.specular.shininess = 0.0; // invalid per existing validate_specular rule
    assert!(validate_material(&m).is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib material:: -- --nocapture`
Expected: FAIL (compile error — `Material`/`validate_material` not yet defined).

- [ ] **Step 3: Implement `Material`/`validate_material` in `src/renderer/material.rs`**

`SpecularMaterial` derives `PartialEq` already or not — check; if not, this task adds `#[derive(PartialEq)]` to `SpecularMaterial` in `texture.rs` (a one-line addition, not a behavior change) so `Material`/`SpecularMaterial` equality comparisons in tests work. `validate_material` calls the existing `super::texture::validate_specular(&material.specular)` first, then checks `base_color`'s three components are finite and `>= 0.0` (mirror `validate_specular`'s own color check exactly).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib material::`
Expected: PASS, 3/3.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/material.rs src/renderer/mod.rs src/renderer/texture.rs
git commit -m "feat(renderer): add Material type bundling texture, base-color tint, and specular"
```

---

## Task 2: Migrate `Object`/`InstanceData`/shaders/call sites to `Material`

**Files:**
- Modify: `src/renderer/scene.rs` (`Object.material: Material` replaces `texture`/`specular`)
- Modify: `src/renderer/pipeline.rs` (`InstanceData` gains `base_color: Vec4`; `From` impl; `instance_attribute_descriptions`)
- Modify: `src/renderer/command.rs` (instance construction; `resolve_texture_set`/`object.specular` call sites)
- Modify: `src/renderer/mod.rs` (validation loop call site)
- Modify: `src/lib.rs` (re-export `Material` alongside existing `SpecularMaterial`/`TextureId`/`Object`)
- Modify: `shaders/triangle.vert`, `shaders/triangle.frag` (new `fragBaseColor` varying; `fragLightSpacePos` location bump; albedo multiply)
- Modify (regenerate only, via `glslc`, not by hand): `shaders/triangle.vert.spv`, `shaders/triangle.frag.spv`, `shaders/triangle_shadow.vert.spv`, `shaders/triangle_shadow.frag.spv`
- Modify: `examples/cube.rs` (migrate `Object` construction to `Material`, neutral tint — no visual change)
- Test: `src/renderer/pipeline.rs`'s existing `#[cfg(test)] mod tests`; `src/renderer/command.rs`'s existing batching tests

**Interfaces:**
- Consumes: Task 1's `Material`/`validate_material`.
- Produces (used by Tasks 4-6): `Object { model: Mat4, mesh: MeshId, material: Material }`; `InstanceData::from((Mat4, Vec3, Vec3, f32))` where the tuple is `(model, base_color, specular_color, shininess)`.

- [ ] **Step 1: Update `pipeline.rs`'s existing instance-data tests to the new shape, watch them fail**

Update `instance_data_converts_rows_and_packs_specular` (rename if you like, e.g. `instance_data_converts_rows_and_packs_base_color_and_specular`) to construct via the 4-tuple and assert a new `base_color: Vec4` field exists with the right values (`xyz` = tint, `w` unused — assert it's whatever the implementation picks, e.g. `1.0`, and pin that choice). Update the byte-offset test (currently asserts `offset_of!(InstanceData, specular) == 64`) to also assert the new field's offset (`80`, since it's appended after `specular`'s 16 bytes). Add a new test asserting `instance_attribute_descriptions()`/the combined `scene_bindings_and_attributes()` attribute list now has 6 instance entries (locations 4-9) instead of 5.

Run: `cargo test --lib pipeline:: -- --nocapture`
Expected: FAIL (compile error — `InstanceData` doesn't have `base_color` yet, `From` impl doesn't accept a 4-tuple).

- [ ] **Step 2: Implement the `InstanceData`/attribute-description changes**

Add `pub base_color: Vec4` as `InstanceData`'s sixth field (after `specular`). Change the `From` impl to `impl From<(Mat4, Vec3, Vec3, f32)> for InstanceData` — destructure as `(model, base_color, specular_color, shininess)`, pack `base_color: Vec4::new(base_color[0], base_color[1], base_color[2], 1.0)` (the `1.0` in `w` is unused padding — pick it for consistency with homogeneous-point conventions elsewhere in the codebase, not for any functional reason). Extend `instance_attribute_descriptions()` to `[vk::VertexInputAttributeDescription; 6]`, adding a sixth entry at `.location(9)` with `offset(std::mem::offset_of!(InstanceData, base_color) as u32)`. Update the doc comment above `InstanceData` (currently says "80 bytes, offsets 0/16/32/48/64" — becomes "96 bytes, offsets 0/16/32/48/64/80").

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --lib pipeline::`
Expected: PASS.

- [ ] **Step 4: Update `scene.rs`'s `Object`, watch the crate fail to build**

Change `Object` to `pub struct Object { pub model: Mat4, pub mesh: MeshId, pub material: Material }`, importing `Material` from `super::material`.

Run: `cargo build --all-targets 2>&1 | head -40`
Expected: FAIL — multiple errors in `mod.rs`, `command.rs`, `examples/cube.rs` referencing `object.texture`/`object.specular`, which no longer exist.

- [ ] **Step 5: Fix every call site**

- `mod.rs`'s per-object validation loop: replace `validate_specular(&object.specular)` with `validate_material(&object.material)`; replace the separate `texture::resolve_texture_set(object.texture, ...)` call — keep it, but read `object.material.texture` instead.
- `command.rs`'s per-frame `InstanceData::from(...)` call: becomes `InstanceData::from((object.model, object.material.base_color, object.material.specular.color, object.material.specular.shininess))`.
- `command.rs`'s `emit_object_draws`' `resolve` closure: `resolve_texture_set(object.texture, ...)` → `resolve_texture_set(object.material.texture, ...)`.
- `examples/cube.rs`: both `Object` literals move their `texture`/`specular` fields into `material: Material { texture: ..., base_color: Vec3::new(1.0, 1.0, 1.0), specular: ... }` (neutral tint — no visual change for this example).

- [ ] **Step 6: Run full build, confirm it compiles**

Run: `cargo build --all-targets`
Expected: PASS, no errors.

- [ ] **Step 7: Add a batching/per-instance-distinctness test, watch it fail, make it pass**

Review Focus: two objects sharing the same `(MeshId, resolved DescriptorSet)` key still batch
into one instanced-draw run per spec 016, but each instance's packed data must stay its own — a
batching implementation that assumed uniform material across a run would silently apply only the
first object's tint to every instance in it. In `command.rs`'s existing test module, add (near
`all_objects_sharing_mesh_and_texture_merge_into_one_run`):

```rust
#[test]
fn batched_objects_keep_their_own_distinct_base_color() {
    // Two objects, same mesh/texture (so they merge into one run per the existing batching
    // test), but Material.base_color differs between them. Build each object's InstanceData the
    // same way command.rs's render path does, and assert instance[0].base_color !=
    // instance[1].base_color — i.e. per-object distinctness survives batching.
}
```
Run: `cargo test --lib command::tests::batched_objects_keep_their_own_distinct_base_color`
Expected: FAIL first (if `InstanceData`'s `From` impl or the per-object construction loop
accidentally shares one value across a run), then PASS once Steps 1-6's implementation is in
place — in practice this should already pass once Step 6 compiles, since `command.rs`'s instance
construction in Step 5 builds one `InstanceData` per object independently; this step's job is to
prove that with a test, not to introduce new production code.

- [ ] **Step 8: Update the shaders**

In `triangle.vert`: add `layout(location = 9) in vec4 inBaseColor;` after `inSpecular`; add `layout(location = 5) flat out vec4 fragBaseColor;` after `fragSpecular`'s `layout(location = 4)`; move the existing `#ifdef SHADOWS_ENABLED` block's `layout(location = 5) out vec4 fragLightSpacePos;` to `layout(location = 6)`; in `main()`, add `fragBaseColor = inBaseColor;` alongside the existing `fragSpecular = inSpecular;` line.

In `triangle.frag`: add `layout(location = 5) flat in vec4 fragBaseColor;`; under its own `#ifdef SHADOWS_ENABLED` block, move `fragLightSpacePos`'s input location from 5 to 6 to match; change the `albedo` line from `texture(tex, fragUV) * vec4(fragColor, 1.0)` to `texture(tex, fragUV) * vec4(fragColor, 1.0) * vec4(fragBaseColor.rgb, 1.0)`.

- [ ] **Step 9: Recompile all four `.spv` outputs**

Run, exactly (per `CLAUDE.md`'s shader-recompilation table):
```bash
glslc shaders/triangle.vert -o shaders/triangle.vert.spv
glslc shaders/triangle.frag -o shaders/triangle.frag.spv
glslc -DSHADOWS_ENABLED shaders/triangle.vert -o shaders/triangle_shadow.vert.spv
glslc -DSHADOWS_ENABLED shaders/triangle.frag -o shaders/triangle_shadow.frag.spv
```
Expected: all four succeed with no `glslc` errors (a location-numbering typo between the vert/frag pair of either variant would otherwise surface here or at pipeline-creation time, not at `cargo build`).

- [ ] **Step 10: Full quality gates and a real-hardware GPU sanity pass**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets
cargo test --all-targets
cargo test --lib -- --ignored
```
Expected: all green, including the full ignored GPU suite (this machine has a real Vulkan driver — see `CLAUDE.md`'s Quality gates section) — this is the cheapest point to catch a shader-location mismatch via an actual pipeline-creation/validation-layer error, well before Task 5's dedicated new goldens.

- [ ] **Step 11: Commit**

```bash
git add src/renderer/scene.rs src/renderer/pipeline.rs src/renderer/command.rs src/renderer/mod.rs src/lib.rs shaders/triangle.vert shaders/triangle.frag shaders/triangle.vert.spv shaders/triangle.frag.spv shaders/triangle_shadow.vert.spv shaders/triangle_shadow.frag.spv examples/cube.rs
git commit -m "feat(renderer): thread Material's base-color tint through InstanceData and the scene shader"
```

---

## Task 3: `base_color_factor` on `LoadedMesh`

**Files:**
- Modify: `src/model.rs`

**Interfaces:**
- Consumes: nothing new from Task 2 (this task is independent — `src/model.rs` has no Vulkan/renderer dependency, per its own design).
- Produces (used by Task 4): `LoadedMesh.base_color_factor: Vec3`.

- [ ] **Step 1: Write the three failing tests**

```rust
#[test]
fn explicit_base_color_factor_is_read() {
    // material.pbrMetallicRoughness.baseColorFactor = [0.2, 0.4, 0.6, 1.0] (no baseColorTexture
    // at all — this is the headline scenario: an untextured, tinted material).
    let model = super::load_gltf(&path).unwrap();
    assert_eq!(model.meshes[0].base_color_factor, Vec3::new(0.2, 0.4, 0.6));
    assert_eq!(model.meshes[0].texture, None);
}

#[test]
fn omitted_base_color_factor_defaults_to_white() {
    // material present (so a baseColorTexture can still be attached), but baseColorFactor field
    // itself is omitted from the JSON — glTF's own default is [1,1,1,1].
    let model = super::load_gltf(&path).unwrap();
    assert_eq!(model.meshes[0].base_color_factor, Vec3::new(1.0, 1.0, 1.0));
}

#[test]
fn no_material_at_all_defaults_base_color_factor_to_white() {
    // primitive has no "material" key at all (already-covered texture: None case from spec 020).
    let model = super::load_gltf(&path).unwrap();
    assert_eq!(model.meshes[0].base_color_factor, Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(model.meshes[0].texture, None);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib model::tests::explicit_base_color_factor_is_read model::tests::omitted_base_color_factor_defaults_to_white model::tests::no_material_at_all_defaults_base_color_factor_to_white`
(Run one at a time if the harness doesn't accept multiple filters — see this plan's own prior art in `src/model.rs`'s test history.)
Expected: FAIL (compile error — `LoadedMesh` has no `base_color_factor` field).

- [ ] **Step 3: Implement**

Add `pub base_color_factor: Vec3` to `LoadedMesh`. In `load_mesh`, read `primitive.material().pbr_metallic_roughness().base_color_factor()` (always returns `[f32; 4]`, defaulting per the `gltf` crate/spec to `[1.0,1.0,1.0,1.0]` when unset) and take its first three components — this one call already handles both "no material at all" (via the crate's existing `Material::default()` fallback — see `src/model.rs`'s existing `primitive.material()` usage) and "material present, factor omitted" identically, so no extra branching is needed beyond what already exists for `texture`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib model::`
Expected: PASS, all tests including the 3 new ones.

- [ ] **Step 5: Commit**

```bash
git add src/model.rs
git commit -m "feat(model): surface glTF baseColorFactor as LoadedMesh::base_color_factor"
```

---

## Task 4: Wire `base_color_factor` into `examples/model.rs`

**Files:**
- Modify: `examples/model.rs`

**Interfaces:**
- Consumes: Task 2's `Material`, Task 3's `LoadedMesh::base_color_factor`.

- [ ] **Step 1: Update the Object-building loop**

Change the `Object` literal built per `loaded_mesh` from `material: Material { texture: ..., base_color: Vec3::new(1.0,1.0,1.0), specular: SpecularMaterial::default() }` (Task 2's placeholder) to `base_color: loaded_mesh.base_color_factor`.

- [ ] **Step 2: Build and confirm no behavior change on the committed asset**

Run: `cargo build --example model`
Expected: compiles clean. (The committed `MechaZiu.glb`'s one material has no explicit `baseColorFactor`, so this is expected to be a no-visual-difference change on the asset already committed — see spec's Interactions section. This step proves the wiring compiles and runs, not that it looks different.)

- [ ] **Step 3: Commit**

```bash
git add examples/model.rs
git commit -m "feat(examples): wire glTF base_color_factor into the model example's materials"
```

---

## Task 5: Headless GPU acceptance goldens

**Files:**
- Modify: `src/renderer/headless.rs`
- Modify: `testdata/golden/README.md`
- Create: `testdata/golden/material_textured_tint.png`, `testdata/golden/material_untextured_tint.png`

**Interfaces:**
- Consumes: Task 2's `Material`/`Object`.

- [ ] **Step 1: Write the two new headless tests**

```rust
#[test]
#[ignore = "needs a real Vulkan driver"]
fn material_textured_tint_matches_golden() {
    // One textured quad with Material { texture: Some(..), base_color: Vec3::new(0.3, 0.6, 1.0),
    // specular: SpecularMaterial::default() } — proves the tint actually reaches the shader
    // (distinguishable from the existing self-test baseline's neutral (1,1,1) tint).
}

#[test]
#[ignore = "needs a real Vulkan driver"]
fn material_untextured_tint_matches_golden() {
    // One UNtextured quad (material.texture: None) with a non-white base_color — proves it no
    // longer renders plain white. This is the concrete gap the spec closes.
}
```
Follow this file's existing scene-setup/golden-comparison pattern exactly (see any existing `..._matches_golden` test for the fixture-building/`assert_matches_golden` call shape).

- [ ] **Step 2: Generate both goldens on real hardware**

Run:
```bash
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless::tests::material_textured_tint_matches_golden -- --ignored --test-threads=1
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless::tests::material_untextured_tint_matches_golden -- --ignored --test-threads=1
```
Expected: both write their PNG and report success.

- [ ] **Step 3: Re-run without the update flag and confirm they now pass as real comparisons**

Run: `cargo test --lib renderer::headless::tests::material_textured_tint_matches_golden renderer::headless::tests::material_untextured_tint_matches_golden -- --ignored`
Expected: PASS.

- [ ] **Step 4: Run the full ignored suite, confirm zero regression on every pre-existing golden**

Run (per `testdata/golden/README.md`'s documented command, with synchronization validation):
```bash
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo test --lib -- --ignored
```
Expected: all tests pass; `git status`/`git diff --stat testdata/golden/` shows only the two new files added, every pre-existing golden PNG untouched.

- [ ] **Step 5: Document the new goldens**

Add a short paragraph to `testdata/golden/README.md` (matching spec 017's/020's own entries there) describing what `material_textured_tint.png`/`material_untextured_tint.png` cover and the regeneration command, per this repo's established convention.

- [ ] **Step 6: Commit**

```bash
git add src/renderer/headless.rs testdata/golden/README.md testdata/golden/material_textured_tint.png testdata/golden/material_untextured_tint.png
git commit -m "test(golden): add textured and untextured base-color-tint acceptance goldens"
```

---

## Task 6: Spec Outcome, ROADMAP checkbox, final gate pass

**Files:**
- Modify: `specs/021-material-system.md` (fill in `## Outcome`)
- Modify: `ROADMAP.md` (check off the Phase 3 "Material system" bullet)

- [ ] **Step 1: Fill in the spec's Outcome section**

Summarize what shipped: the `Material` shape, the `InstanceData`/shader location changes, the real gap closed (untextured tinted glTF materials), and confirmation that every pre-existing golden remained byte-identical.

- [ ] **Step 2: Check off ROADMAP.md**

Change:
```
- [ ] **Material system**: decouple "what a shader needs" from "what an asset provides"
```
to:
```
- [x] **Material system**: `Material { texture, base_color, specular }` on `Object`; glTF's
      `baseColorFactor` now surfaces through `LoadedMesh` — [spec 021](specs/021-material-system.md).
      Metallic/roughness/emissive/normal maps/alpha remain a future PBR-materials spec.
```

- [ ] **Step 3: Run the full quality gate suite one last time**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets
cargo test --all-targets
```
Expected: all four exit 0.

- [ ] **Step 4: Commit**

```bash
git add specs/021-material-system.md ROADMAP.md
git commit -m "docs(spec): record spec 021 outcome; check off material system in ROADMAP"
```
