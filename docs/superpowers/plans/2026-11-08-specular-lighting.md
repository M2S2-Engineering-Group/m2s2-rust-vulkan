# Blinn-Phong Specular Lighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add camera-dependent Blinn-Phong specular highlights on top of the existing directional diffuse light, with a per-object `SpecularMaterial { color, shininess }` and an explicit `Camera { view, proj, position }` replacing the separate `view`/`proj` render arguments.

**Architecture:** Task 1 lands all data-model/layout/validation plumbing — `Camera`, `SpecularMaterial`, the widened camera UBO (144 bytes) and model push constant (80 bytes), both now visible to the fragment stage — while the shaders are untouched, so the render is byte-for-byte unchanged (same regression-proving pattern as specs 010/011). Task 2 updates the shaders to compute the highlight and wires a visible on/off demonstration into the cube example.

**Tech Stack:** Rust, `ash` (Vulkan 1.3), `bytemuck`, `m2s2-math`; GLSL compiled offline with `glslc`, SPIR-V committed, validated with `spirv-val`.

**Spec:** `specs/013-specular-lighting.md`

## Global Constraints

- All lighting vectors are **world space**. `N`/world position come from the vertex shader; `L` from the existing directional light; `V` from `camera.position - worldPos`; `H = normalize(L + V)`.
- **No PBR/energy conservation, no point/spot lights, no shadows, no normal maps, no specular textures, no new mesh/resource ownership.** Educational extension only.
- **Camera UBO** grows to 144 bytes: `view`(0,64) `proj`(64,64) `position`(128, vec3) `_pad`(140, f32); set 0 binding 0 becomes visible to `VERTEX | FRAGMENT`.
- **Model push constant** grows to 80 bytes: `model`(0,64) `specular_color`(64,12) `shininess`(76,4); one range, offset 0, `VERTEX | FRAGMENT`. Both already fit well under the Vulkan-guaranteed 128-byte minimum `maxPushConstantsSize`.
- **Default material** (`SpecularMaterial::default()`): `color (0,0,0)`, `shininess 32.0` — zero highlight, so `render_frame` and any object that doesn't opt in reproduce today's diffuse-only look exactly.
- Validate camera position (finite) and every object's specular material (color finite & nonnegative, shininess finite & ≥ 1.0) before any acquire/mapped-write/fence-reset; invalid input submits nothing.
- Quality gates zero-warning at every commit: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test --all-targets`, `cargo test --doc`.
- Conventional Commits; **no `Co-Authored-By`/AI-attribution trailer**.
- **GPU verification is a manual visual check** (resolved decision, spec Outcome) — no pixel-tolerance/image-comparison infrastructure.

## Review Focus

- **Non-finite camera position** (`NaN`/`±inf` in any component) → `VulkanError::ValidationError` before any GPU work. *Test in Task 1 (`validate_camera`).*
- **Invalid specular material** (negative/non-finite color, or shininess `< 1.0`/non-finite) → `ValidationError`; boundary `shininess == 1.0` is valid. *Test in Task 1 (`validate_specular`).*
- **Degenerate shader vectors** (camera exactly at the surface point, or `L + V` near zero on a grazing/back-facing angle) → highlight must cleanly zero out, never `NaN`/garbage. *Shader-only; no CPU test possible — exercised in Task 2's GPU acceptance check.*
- **Specular-off (zero color) must reproduce the exact pre-013 diffuse look**, no accidental tint/offset from the added terms. *Task 2 GPU acceptance.*
- **`render_frame`'s implicit default material must not change the triangle example's output** — the simplest API call stays visually identical to pre-013. *Task 2 GPU acceptance.*

---

### Task 1: Camera, SpecularMaterial, and layout plumbing (shaders untouched — render unchanged)

Lands every data type, GPU-layout, and validation change, and migrates both examples to the new render-call signature. The old shaders don't read any of the new bytes, so nothing visually changes — this is the regression gate for the plumbing, exactly as specs 010/011 proved their own layout changes before touching shaders.

**Files:**
- Modify: `src/renderer/uniform.rs` (`Camera` public struct; `CameraUniforms` grows to 144 bytes; `From<&Camera>`; `validate_camera`; `Uniforms::new`'s stage bumps to `VERTEX | FRAGMENT`)
- Modify: `src/renderer/texture.rs` (`SpecularMaterial` public struct + `Default`; `validate_specular`; `Object` gains `specular: SpecularMaterial`)
- Modify: `src/renderer/pipeline.rs` (`ModelPushConstants` grows to 80 bytes; `MODEL_PUSH_STAGES` becomes `VERTEX | FRAGMENT`; test updated)
- Modify: `src/renderer/command.rs` (`emit_object_draws`'s push-constant construction includes the material; existing test literals updated)
- Modify: `src/renderer/mod.rs` (`render_frame`/`render_objects`/`render_objects_with_text` take `uniform::Camera` instead of `view: Mat4, proj: Mat4`; validate camera + every object's specular before any GPU work; build `CameraUniforms::from(&camera)`)
- Modify: `src/lib.rs` (re-export `Camera`, `SpecularMaterial`)
- Modify: `examples/cube.rs`, `examples/triangle.rs` (construct `Camera { view, proj, position }`; cube's two objects get `specular: SpecularMaterial::default()` for now — Task 2 gives them real materials)

**Interfaces:**
- Consumes: existing `Uniforms`/`PerFrameUniform`, `ModelPushConstants`/`model_push_range`/`validate_push_limit`, `Object`, `draw_frame`'s unchanged `camera: &CameraUniforms` parameter.
- Produces:
  - `pub struct Camera { pub view: Mat4, pub proj: Mat4, pub position: Vec3 }` (`Copy, Clone, Debug`), in `uniform.rs`.
  - `pub(crate) fn validate_camera(camera: &Camera) -> Result<()>` — checks only `position` is finite component-wise (view/proj validity remains the caller's existing contract, per spec).
  - `impl From<&Camera> for CameraUniforms` (adds `_pad: 0.0`).
  - `pub struct SpecularMaterial { pub color: Vec3, pub shininess: f32 }` (`Copy, Clone, Debug`) + `impl Default` (`color (0,0,0)`, `shininess 32.0`), in `texture.rs`.
  - `pub(crate) fn validate_specular(material: &SpecularMaterial) -> Result<()>`.
  - `Object { pub model: Mat4, pub texture: Option<TextureId>, pub specular: SpecularMaterial }`.
  - `render_frame(&mut self, model: Mat4, camera: uniform::Camera) -> Result<()>`, `render_objects(&mut self, objects: &[Object], camera: uniform::Camera) -> Result<()>`, `render_objects_with_text(&mut self, objects: &[Object], camera: uniform::Camera, text: &[text::TextDraw<'_>]) -> Result<()>`.

- [ ] **Step 1: Write the failing `CameraUniforms`/`Camera` tests**

In `src/renderer/uniform.rs` tests, replace `camera_uniforms_match_shader_layout` and add two tests:

```rust
#[test]
fn camera_uniforms_match_shader_layout() {
    assert_eq!(std::mem::size_of::<CameraUniforms>(), 144);
    assert_eq!(std::mem::offset_of!(CameraUniforms, view), 0);
    assert_eq!(std::mem::offset_of!(CameraUniforms, proj), 64);
    assert_eq!(std::mem::offset_of!(CameraUniforms, position), 128);
    assert_eq!(std::mem::offset_of!(CameraUniforms, _pad), 140);
}

#[test]
fn camera_converts_fields_and_zeroes_padding() {
    let camera = Camera { view: Mat4::IDENTITY, proj: Mat4::IDENTITY, position: Vec3::new(1.0, 2.0, 3.0) };
    let u = CameraUniforms::from(&camera);
    assert_eq!((u.view, u.proj, u.position, u._pad), (camera.view, camera.proj, camera.position, 0.0));
}

#[test]
fn validate_camera_rejects_non_finite_position() {
    let base = Camera { view: Mat4::IDENTITY, proj: Mat4::IDENTITY, position: Vec3::new(0.0, 0.0, 0.0) };
    assert!(validate_camera(&base).is_ok());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for i in 0..3 {
            let mut c = base;
            c.position[i] = bad;
            assert!(validate_camera(&c).is_err());
        }
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib renderer::uniform`
Expected: FAIL — `Camera`/`validate_camera` don't exist; `CameraUniforms` is still 128 bytes with no `position`/`_pad`.

- [ ] **Step 3: Implement `Camera`, the widened `CameraUniforms`, `From`, and `validate_camera`**

In `src/renderer/uniform.rs`, add `use crate::error::VulkanError;` and `use crate::math::Vec3;` to the existing imports. Add the `Camera` struct and `validate_camera` above `CameraUniforms`. Add `position: Vec3, _pad: f32` to `CameraUniforms` (after `proj`) and the `From<&Camera>` impl. Change `Uniforms::new`'s `PerFrameUniform::new(..., vk::ShaderStageFlags::VERTEX)` call to `vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT`.

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --lib renderer::uniform`
Expected: PASS.

- [ ] **Step 5: Write the failing `SpecularMaterial`/`validate_specular` tests**

In `src/renderer/texture.rs` tests (create a `#[cfg(test)] mod tests` if none exists yet — check first):

```rust
#[test]
fn specular_material_default_matches_pre_013_look() {
    let m = SpecularMaterial::default();
    assert_eq!((m.color, m.shininess), (Vec3::new(0.0, 0.0, 0.0), 32.0));
}

#[test]
fn validate_specular_rejects_invalid_color_and_shininess() {
    let base = SpecularMaterial::default();
    assert!(validate_specular(&base).is_ok());
    for bad in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for i in 0..3 {
            let mut m = base;
            m.color[i] = bad;
            assert!(validate_specular(&m).is_err());
        }
    }
    for bad_shininess in [0.0, 0.5, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(validate_specular(&SpecularMaterial { color: base.color, shininess: bad_shininess }).is_err());
    }
    assert!(validate_specular(&SpecularMaterial { color: base.color, shininess: 1.0 }).is_ok());
}
```

- [ ] **Step 6: Run to verify they fail**

Run: `cargo test --lib renderer::texture`
Expected: FAIL — `SpecularMaterial`/`validate_specular` don't exist.

- [ ] **Step 7: Implement `SpecularMaterial`, `Default`, `validate_specular`, and `Object.specular`**

In `src/renderer/texture.rs`, add `use crate::math::Vec3;` to the existing `math::Mat4` import. Add `SpecularMaterial` (`Copy, Clone, Debug`), its `Default`, and `validate_specular` (color: all three components finite and `>= 0.0`; shininess: finite and `>= 1.0`). Add `pub specular: SpecularMaterial` to `Object`.

- [ ] **Step 8: Run to verify they pass (and see the compile breaks `Object` causes)**

Run: `cargo test --lib renderer::texture`
Expected: the two new tests PASS; the crate as a whole now fails to build (every `Object { .. }` literal is missing `specular`) — expected, fixed in the next steps.

- [ ] **Step 9: Write the failing `ModelPushConstants` layout test**

In `src/renderer/pipeline.rs` tests, replace `model_payload_and_range_match_shader_layout`:

```rust
#[test]
fn model_payload_and_range_match_shader_layout() {
    assert_eq!(std::mem::size_of::<ModelPushConstants>(), 80);
    assert_eq!(std::mem::offset_of!(ModelPushConstants, model), 0);
    assert_eq!(std::mem::offset_of!(ModelPushConstants, specular_color), 64);
    assert_eq!(std::mem::offset_of!(ModelPushConstants, shininess), 76);
    let range = model_push_range();
    assert_eq!(range.offset, 0);
    assert_eq!(range.size, 80);
    assert_eq!(range.stage_flags, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
}
```

(`push_limit_accepts_equal_or_larger_and_rejects_smaller` needs no code change — it already derives its expectations from `MODEL_PUSH_SIZE`, which will now be 80.)

- [ ] **Step 10: Run to verify it fails**

Run: `cargo test --lib renderer::pipeline`
Expected: FAIL — `ModelPushConstants` is still 64 bytes with no `specular_color`/`shininess`.

- [ ] **Step 11: Extend `ModelPushConstants` and `MODEL_PUSH_STAGES`**

In `src/renderer/pipeline.rs`, add `use crate::math::Vec3;` alongside the existing `Mat4` import. Add `pub specular_color: Vec3, pub shininess: f32` to `ModelPushConstants` (after `model` — this naturally lands at offsets 64/76 with no gaps, since `Mat4` is a clean 64-byte multiple and `Vec3`'s 16-byte GLSL alignment requirement is satisfied by offset 64). Change `MODEL_PUSH_STAGES` to `vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT`.

- [ ] **Step 12: Run to verify the pipeline tests pass**

Run: `cargo test --lib renderer::pipeline`
Expected: PASS.

- [ ] **Step 13: Fix the remaining compile breaks — `command.rs` push construction, test literals, `mod.rs` signatures, `lib.rs`, both examples**

In `src/renderer/command.rs`:
- `emit_object_draws`'s `ModelPushConstants { model: object.model }` becomes `ModelPushConstants { model: object.model, specular_color: object.specular.color, shininess: object.specular.shininess }`.
- In the test module, add `specular: SpecularMaterial::default()` / `specular: SpecularMaterial { color: Vec3::new(1.0, 1.0, 1.0), shininess: 64.0 }` to the two `Object` literals in `mixed_objects_bind_resolved_set_then_push_then_draw_in_order`, and extend both expected `ModelPushConstants { .. }` literals with matching `specular_color`/`shininess` fields. Import `SpecularMaterial` (`use super::texture::SpecularMaterial;` or qualify inline).

In `src/renderer/mod.rs`:
- Add `validate_specular` and `SpecularMaterial` to the existing `use texture::{Object, StoredTexture};` import line.
- Change `render_frame`/`render_objects`/`render_objects_with_text` to the signatures in Interfaces above. `render_frame` builds `Object { model, texture: None, specular: texture::SpecularMaterial::default() }` — adjust the import so `SpecularMaterial` is reachable, then call `self.render_objects(&[that object], camera)`.
- At the top of `render_objects_with_text`'s body, right after `self.ensure_not_terminal()?;`, add `uniform::validate_camera(&camera)?;` then `for object in objects { texture::validate_specular(&object.specular)?; }` — before the resize block, before any acquire/mapped write.
- Where the function currently does `let camera = uniform::CameraUniforms { view, proj };`, change to `let camera_uniforms = uniform::CameraUniforms::from(&camera);` and pass `&camera_uniforms` into `draw_frame` (the outer parameter is now named `camera: uniform::Camera`, so the local conversion needs a different name).

In `src/lib.rs`: change `pub use renderer::texture::{Object, TextureId};` to `pub use renderer::texture::{Object, SpecularMaterial, TextureId};`, and add `pub use renderer::uniform::Camera;`.

In `examples/cube.rs` and `examples/triangle.rs`: add `Camera, SpecularMaterial` to the `use m2s2_rust_vulkan::{...}` import line. Replace the `render_frame(model, view, proj)` / `render_objects_with_text(&objects, view, proj, text)` calls with `Camera { view, proj, position: camera_position }` (cube already computes `camera_position`; triangle's fixed camera position is `Vec3::new(0.0, 0.0, 2.0)` — reuse the literal already passed to `look_at`). In `cube.rs`, add `specular: SpecularMaterial::default()` to both `Object` literals (Task 2 gives them real materials).

- [ ] **Step 14: Run every gate**

Run: `cargo fmt --all && cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets && cargo test --doc`
Expected: all clean. (The old shaders don't declare a `position`/material block, so nothing GPU-visible changes yet — valid Vulkan: granting a stage push-constant/descriptor access it doesn't use is harmless.)

- [ ] **Step 15: GPU regression — render must be unchanged**

Run (outside the sandbox, on a machine with the Vulkan driver): `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo run --example cube` (and `triangle`).
Expected: both render **exactly as before this task** (no visible change — the shaders don't read the new bytes yet), zero validation errors.

- [ ] **Step 16: Commit**

```bash
git add src/renderer/uniform.rs src/renderer/texture.rs src/renderer/pipeline.rs \
        src/renderer/command.rs src/renderer/mod.rs src/lib.rs \
        examples/cube.rs examples/triangle.rs
git commit -m "feat(renderer)!: add Camera and SpecularMaterial; widen camera/model GPU layouts"
```

---

### Task 2: Blinn-Phong shaders and the cube's highlight demonstration

Updates the shaders to compute the highlight and gives the cube example a visible, toggleable specular on/off comparison.

**Files:**
- Modify: `shaders/triangle.vert` (+ committed `.spv`)
- Modify: `shaders/triangle.frag` (+ committed `.spv`)
- Modify: `examples/cube.rs` (real specular materials; an on/off toggle key)

**Interfaces:**
- Consumes: `Camera.position` (via the UBO), `ModelPushConstants.{specular_color,shininess}` (via the push range), the existing world-space `fragNormal`.
- Produces: no new Rust API — this task is shader-only plus the example demonstration.

- [ ] **Step 1: Add the world-position varying, and stop pre-normalizing the normal, in the vertex shader**

In `shaders/triangle.vert`, add `layout(location = 3) out vec3 fragWorldPos;`. In `main`, compute `vec4 worldPos = object.model * vec4(inPosition, 1.0);`, use `worldPos` for `gl_Position` (`camera.proj * camera.view * worldPos`), and set `fragWorldPos = worldPos.xyz;`. The existing `CameraUniforms`/`ModelPushConstants` block declarations in this shader are unchanged — the vertex stage never reads `camera.position`.

Also change `fragNormal = normalize(normalMatrix * inNormal);` to `fragNormal = normalMatrix * inNormal;` (drop the `normalize`). Per spec: interpolating two already-unit vectors across a triangle doesn't yield a unit vector at each pixel anyway, and pre-normalizing discards the relative-magnitude information non-uniform scale encodes in the transformed normal — normalize once, after interpolation, in the fragment shader (which already does `vec3 N = normalize(fragNormal);` and needs no change for this).

- [ ] **Step 2: Add the highlight computation to the fragment shader**

In `shaders/triangle.frag`, add `layout(location = 3) in vec3 fragWorldPos;`. Add a minimal camera block that reads only the position (explicit offset, no `row_major` — there's no matrix member):

```glsl
layout(set = 0, binding = 0) uniform CameraUniforms {
    layout(offset = 128) vec3 position;
} camera;
```

Add the material push-constant subset (also explicit offsets, no `row_major`):

```glsl
layout(push_constant) uniform MaterialPushConstants {
    layout(offset = 64) vec3 specularColor;
    layout(offset = 76) float shininess;
} material;
```

Replace the final `outColor = vec4(shade, 1.0) * albedo;` line with the guarded Blinn-Phong term (named minimum squared length `1e-12` per the spec, to avoid normalizing a degenerate vector):

```glsl
const float MIN_SAFE_LENGTH_SQUARED = 1e-12;
...
vec3 highlight = vec3(0.0);
vec3 toCamera = camera.position - fragWorldPos;
float viewLenSq = dot(toCamera, toCamera);
if (diff > 0.0 && viewLenSq > MIN_SAFE_LENGTH_SQUARED) {
    vec3 V = toCamera * inversesqrt(viewLenSq);
    if (dot(N, V) > 0.0) {
        vec3 halfway = L + V;
        float halfLenSq = dot(halfway, halfway);
        if (halfLenSq > MIN_SAFE_LENGTH_SQUARED) {
            vec3 H = halfway * inversesqrt(halfLenSq);
            highlight = light.color * material.specularColor * pow(max(dot(N, H), 0.0), material.shininess);
        }
    }
}
outColor = vec4(shade, 1.0) * albedo + vec4(highlight, 0.0);
```

(`diff`, `N`, `L`, `shade`, `albedo` are the existing locals from spec 011 — unchanged. The highlight is added after the albedo multiply, per spec: "Do not tint the highlight by diffuse albedo or vertex color.")

- [ ] **Step 3: Recompile and validate**

```bash
glslc shaders/triangle.vert -o shaders/triangle.vert.spv
glslc shaders/triangle.frag -o shaders/triangle.frag.spv
spirv-val shaders/triangle.vert.spv
spirv-val shaders/triangle.frag.spv
```

- [ ] **Step 4: Verify CPU gates**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets && cargo test --doc`
Expected: all clean; `renderer::pipeline::tests::*_spirv_parses_and_starts_with_magic` pass against the recompiled binaries.

- [ ] **Step 5: Give the cube real specular materials with an on/off toggle**

In `examples/cube.rs`, add a `highlights_enabled: bool` (default `true`) alongside `show_diagnostics`, toggled by pressing **H** (same pattern as the existing F3 toggle). In the `AboutToWait` arm, build each object's `specular` from `highlights_enabled`:

```rust
let specular = if highlights_enabled {
    [
        SpecularMaterial { color: Vec3::new(1.0, 1.0, 1.0), shininess: 16.0 },
        SpecularMaterial { color: Vec3::new(1.0, 1.0, 1.0), shininess: 64.0 },
    ]
} else {
    [SpecularMaterial::default(), SpecularMaterial::default()]
};
let objects = [
    Object { model: models[0], texture: Some(tex_a), specular: specular[0] },
    Object { model: models[1], texture: Some(tex_b), specular: specular[1] },
];
```

This gives the two cubes identical specular color with different shininess (broad vs. sharp highlight) when enabled, and exactly today's diffuse-only look (zero color) when disabled — the spec's required off/on comparison, driven by one key rather than extra geometry.

- [ ] **Step 6: Verify gates**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 7: GPU acceptance (manual — resolved decision, no pixel-tolerance infra)**

Run (outside the sandbox): `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo run --example cube`
Confirm, by eye: with highlights on, each cube shows a highlight that moves with the camera under the stationary sun — cube 0 (shininess 16) broader, cube 1 (shininess 64) sharper and tighter. Pressing **H** turns highlights off and the cubes look exactly as they did before spec 013 (diffuse + texture only, no tint). No validation/synchronization errors across resize, minimize/restore, and shutdown. Also run `cargo run --example triangle` and confirm it looks unchanged (default material, zero highlight).

- [ ] **Step 8: Record the outcome and roadmap**

In `specs/013-specular-lighting.md`, fill the Outcome section (what landed, test counts, gates, the GPU acceptance result — report pass/fail per the manual check in Step 7, not a pixel tolerance). In `ROADMAP.md`, extend the "Basic lighting" note: ambient+diffuse+specular+normals are now done; point/spot lights and multiple lights remain (roadmap box stays unchecked until those land).

- [ ] **Step 9: Commit**

```bash
git add shaders/triangle.vert shaders/triangle.vert.spv shaders/triangle.frag shaders/triangle.frag.spv \
        examples/cube.rs specs/013-specular-lighting.md ROADMAP.md
git commit -m "feat(renderer): Blinn-Phong specular highlights and cube on/off demonstration"
```
