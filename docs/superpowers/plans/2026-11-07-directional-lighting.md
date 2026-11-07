# Directional Lighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Light scene objects with a single stationary directional ("sun") light using per-pixel Lambert diffuse + ambient shading, combined with the existing texture × vertex-color.

**Architecture:** Surface normals become a `Vertex` attribute. The camera uniform's per-frame mapped-UBO machinery is factored into a shared helper so a second per-frame uniform — the light — reuses it at a new descriptor **set 2** (fragment stage). The renderer stores a `DirectionalLight` (defaulting to a sensible sun) and uploads it each frame; the fragment shader shades `(ambient + color·max(dot(N,L),0)) × albedo`. The free-fly camera, reset key, and 24-vertex per-face cube are example-only.

**Tech Stack:** Rust, `ash` (Vulkan 1.3), `gpu-allocator`, `bytemuck`, `m2s2-math`; GLSL compiled offline with `glslc`, SPIR-V committed; `winit` (examples only).

**Spec:** `specs/011-directional-lighting.md`

## Global Constraints

- Vulkan **1.3** required (spec 009); dynamic rendering is core. Do not reintroduce extension loaders.
- Matrices are **row-major** (`m2s2-math`); GLSL uniform/push blocks already declare `row_major`.
- Light UBO block is **std140**: three `vec3`s at offsets **0 / 16 / 32**, block size **48**.
- Camera at **set 0** (vertex), per-object texture at **set 1** (fragment, spec 010), **light at set 2, binding 0, `UNIFORM_BUFFER`, fragment stage**.
- **No new crate dependencies.** No specular, no point/spot lights, no multiple lights, no shadows (all deferred).
- Quality gates must pass with **zero warnings** at every commit: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test --all-targets`.
- Commit subjects follow **Conventional Commits**. **No `Co-Authored-By` / AI-attribution trailer** in any commit.
- Shader edits compiled offline and the `.spv` committed: `glslc shaders/triangle.vert -o shaders/triangle.vert.spv` (and `.frag`), then `spirv-val` each.

## Review Focus

- **A model with non-uniform scale** → normals must stay perpendicular to the surface. The vertex shader uses the inverse-transpose normal matrix `transpose(inverse(mat3(model)))`, not `mat3(model)`. *No CPU test can exercise a shader; verify by reading the recompiled vertex shader in Task 3 and by the GPU run.*
- **A degenerate vertex normal `(0,0,0)`** → `normalize` yields NaN and the fragment goes garbage. This is caller error (consumer supplies normals); documented in the `Vertex` doc comment, not guarded in-shader. *Reviewer note only; GPU-visible.*
- **`set_directional_light` called between frames while the GPU reads an earlier frame** → no read-of-in-flight-data race, because the light is per-frame-in-flight and written only after the frame's fence wait (same discipline as the camera). *Structural guarantee; verified by the Task 3 GPU run with synchronization validation.*
- **An empty `&[Object]` slice** → the frame still clears, binds set 0 and set 2, and presents with no object draws. *Covered by the existing `empty_objects_emit_no_commands` test (unchanged) in Task 3; GPU-confirmed.*
- **No `set_directional_light` call at all** → objects are still lit by a sensible default sun. *Covered by the `DirectionalLight::default` test in Task 3.*

---

### Task 1: Add `normal` to the `Vertex` type

Adds a surface-normal attribute to the public vertex format and feeds a value from both examples. Shaders are left untouched this task — a vertex attribute the shader doesn't read is valid Vulkan, so the render is unchanged. Isolates the public-ABI change from the lighting work (mirrors spec 010's `uv` task).

**Files:**
- Modify: `src/renderer/buffer.rs` (the `Vertex` struct, `attribute_descriptions`, doc comment, packing + attribute tests)
- Modify: `examples/cube.rs` (add `normal` to the 8 vertices)
- Modify: `examples/triangle.rs` (add `normal` to the 3 vertices)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Vertex { position: Vec3, color: Vec3, uv: Vec2, normal: Vec3 }`; `Vertex::attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4]` with a 4th entry `location = 3`, `R32G32B32_SFLOAT`, `offset_of!(Vertex, normal)`.

- [ ] **Step 1: Update the packing + attribute tests to expect `normal`**

In `src/renderer/buffer.rs` tests, update `vertex_is_tightly_packed` and `attribute_descriptions_map_position_color_then_uv` (rename the latter to `..._then_uv_then_normal`):

```rust
#[test]
fn vertex_is_tightly_packed() {
    assert_eq!(std::mem::size_of::<Vertex>(), 44);
    assert_eq!(std::mem::offset_of!(Vertex, position), 0);
    assert_eq!(std::mem::offset_of!(Vertex, color), 12);
    assert_eq!(std::mem::offset_of!(Vertex, uv), 24);
    assert_eq!(std::mem::offset_of!(Vertex, normal), 32);
}

#[test]
fn attribute_descriptions_map_position_color_then_uv_then_normal() {
    let attrs = Vertex::attribute_descriptions();
    assert_eq!(attrs.len(), 4);
    assert_eq!(attrs[3].location, 3);
    assert_eq!(attrs[3].format, vk::Format::R32G32B32_SFLOAT);
    assert_eq!(attrs[3].offset, 32);
    // existing location 0/1/2 assertions stay unchanged
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib renderer::buffer`
Expected: FAIL — `normal` field does not exist / `attribute_descriptions` returns 3 entries.

- [ ] **Step 3: Add the `normal` field and fourth attribute**

In `src/renderer/buffer.rs`, add `normal: Vec3` after `uv` in the `Vertex` struct, update the doc comment to mention the surface normal (and note a `(0,0,0)` normal is caller error → NaN shading), and add a fourth `vk::VertexInputAttributeDescription` with `.location(3).binding(0).format(vk::Format::R32G32B32_SFLOAT).offset(std::mem::offset_of!(Vertex, normal) as u32)`, changing the return type to `[_; 4]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib renderer::buffer`
Expected: PASS.

- [ ] **Step 5: Add a normal to both examples' vertices so they compile**

In `examples/triangle.rs`, add `normal: Vec3::new(0.0, 0.0, 1.0)` (facing the camera) to each of the 3 vertices. In `examples/cube.rs`, add a `normal` to each of the 8 vertices — a placeholder `Vec3::new(0.0, 0.0, 0.0)` is **not** acceptable (NaN); use the normalized position (`Vec3::new(x, y, z)` normalized) so it compiles and renders sanely. The real per-face 24-vertex rebuild is Task 4.

- [ ] **Step 6: Verify the whole workspace builds and gates pass**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean. (The vertex shader not reading `location = 3` is a valid unused vertex attribute.)

- [ ] **Step 7: Commit**

```bash
git add src/renderer/buffer.rs examples/cube.rs examples/triangle.rs
git commit -m "feat(renderer): add normal attribute to Vertex"
```

---

### Task 2: Extract a shared per-frame mapped-uniform helper

Factors the per-frame-in-flight mapped-UBO machinery out of `uniform.rs` so the light (Task 3) reuses it instead of duplicating it (shared-before-one-off). Pure refactor: the camera path's public surface and behavior are unchanged, guarded by the existing camera tests staying green.

**Files:**
- Create: `src/renderer/per_frame_uniform.rs`
- Modify: `src/renderer/uniform.rs` (`Uniforms` becomes a thin wrapper)
- Modify: `src/renderer/mod.rs` (add `pub(crate) mod per_frame_uniform;` to the module list)

**Interfaces:**
- Consumes: `Buffer::new_host_visible_mapped`, `Cleanup`, `FRAMES_IN_FLIGHT`.
- Produces:
  - `pub(crate) struct PerFrameUniform` owning the set layout, pool, `[DescriptorSet; FRAMES_IN_FLIGHT]`, and `[Buffer; FRAMES_IN_FLIGHT]`.
  - `PerFrameUniform::new(device: &VulkanDevice, allocator: &mut Allocator, size: u64, binding: u32, stage: vk::ShaderStageFlags) -> Result<Self>`
  - `update(&self, frame: usize, bytes: &[u8])`, `descriptor_set_layout(&self) -> vk::DescriptorSetLayout`, `descriptor_set(&self, frame: usize) -> vk::DescriptorSet`, `destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator)`
  - Unchanged: `Uniforms::{new, update(&self, frame, &CameraUniforms), descriptor_set_layout, descriptor_set, destroy}` and `CameraUniforms`.

- [ ] **Step 1: Confirm the camera tests pass before refactoring**

Run: `cargo test --lib renderer::uniform`
Expected: PASS (`camera_uniforms_match_shader_layout`). This is the green baseline the refactor must preserve.

- [ ] **Step 2: Create `per_frame_uniform.rs` with the generalized helper**

Move the body of the current `Uniforms::new` into `PerFrameUniform::new`, replacing: the hardcoded `UNIFORM_BUFFER`/`VERTEX`/`UBO_BINDING` with the `binding`/`stage` parameters, and `size_of::<CameraUniforms>()` with the `size` parameter. Keep the existing `Cleanup`-guarded rollback and the per-frame buffer/descriptor-set allocation verbatim. `update` writes `bytes` to `buffers[frame]` via `write_mapped`; `destroy` frees buffers then pool+layout. Add `pub(crate) mod per_frame_uniform;` to `src/renderer/mod.rs`.

- [ ] **Step 3: Reduce `Uniforms` to a wrapper over `PerFrameUniform`**

In `src/renderer/uniform.rs`, keep `CameraUniforms`, `UBO_BINDING`, and the layout test. Replace the struct body with a single `inner: PerFrameUniform` field. `Uniforms::new` calls `PerFrameUniform::new(device, allocator, std::mem::size_of::<CameraUniforms>() as u64, UBO_BINDING, vk::ShaderStageFlags::VERTEX)`. `update(&self, frame, ubo: &CameraUniforms)` forwards `bytemuck::bytes_of(ubo)`; the three accessors/`destroy` forward to `inner`.

- [ ] **Step 4: Verify camera tests and full gates still pass**

Run: `cargo test --lib renderer::uniform && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean; `camera_uniforms_match_shader_layout` still passes. No behavior change.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/per_frame_uniform.rs src/renderer/uniform.rs src/renderer/mod.rs
git commit -m "refactor(renderer): extract shared per-frame uniform helper"
```

---

### Task 3: Directional light — set 2, shaders, and the renderer light API

Adds the light resource, the public `DirectionalLight` + `set_directional_light`, the set-2 bind, and the shading shaders, all in one reviewable unit (anything less leaves dead code that fails `-D warnings`). After this task the cube is visibly lit (smoothly, via Task 1's placeholder normals); Task 4 makes it flat per-face.

**Files:**
- Create: `src/renderer/light.rs`
- Modify: `src/renderer/pipeline.rs` (`new`/`create` receive the three-layout slice — no signature change; callers pass 3)
- Modify: `src/renderer/command.rs` (`draw_frame`/`record` take the light + its set; bind set 2; write the light buffer)
- Modify: `src/renderer/mod.rs` (module list; own `Light` + current `DirectionalLight`; `set_directional_light`; construction/resize/teardown; thread into the frame)
- Modify: `src/lib.rs` (re-export `DirectionalLight`)
- Modify: `shaders/triangle.vert`, `shaders/triangle.frag` (+ committed `.spv`)

**Interfaces:**
- Consumes: `PerFrameUniform` (Task 2); `Vertex.normal` (Task 1); existing `GraphicsPipeline`, `FrameContext::{draw_frame,record}`, `CameraUniforms`, the model push constant, `Object`, texture sets.
- Produces:
  - `pub struct DirectionalLight { pub direction: Vec3, pub color: Vec3, pub ambient: Vec3 }` (`Copy, Clone, Debug`), with `impl Default`.
  - internal `LightUniforms` (`#[repr(C)]`, `Pod`, `Zeroable`): `direction: Vec3, _pad0: f32, color: Vec3, _pad1: f32, ambient: Vec3, _pad2: f32`, with `impl From<&DirectionalLight>`.
  - `pub(crate) struct Light` wrapping `PerFrameUniform`: `new(device, allocator) -> Result<Self>`, `update(&self, frame, &DirectionalLight)`, `descriptor_set_layout`, `descriptor_set(frame)`, `destroy`.
  - `VulkanRenderer::set_directional_light(&mut self, light: DirectionalLight)`.

- [ ] **Step 1: Write failing tests for the light layout, conversion, and default**

Create `src/renderer/light.rs` with its test module:

```rust
#[test]
fn light_uniforms_match_std140_layout() {
    assert_eq!(std::mem::size_of::<LightUniforms>(), 48);
    assert_eq!(std::mem::offset_of!(LightUniforms, direction), 0);
    assert_eq!(std::mem::offset_of!(LightUniforms, color), 16);
    assert_eq!(std::mem::offset_of!(LightUniforms, ambient), 32);
}

#[test]
fn directional_light_converts_fields_and_zeroes_padding() {
    let light = DirectionalLight {
        direction: Vec3::new(1.0, -2.0, 3.0),
        color: Vec3::new(0.4, 0.5, 0.6),
        ambient: Vec3::new(0.1, 0.1, 0.1),
    };
    let u = LightUniforms::from(&light);
    assert_eq!(u.direction, light.direction);
    assert_eq!(u.color, light.color);
    assert_eq!(u.ambient, light.ambient);
    assert_eq!((u._pad0, u._pad1, u._pad2), (0.0, 0.0, 0.0));
}

#[test]
fn default_light_is_a_sun_with_ambient_floor() {
    let d = DirectionalLight::default();
    // a non-zero travel direction, non-black diffuse, and a small non-zero ambient floor
    assert!(d.direction != Vec3::new(0.0, 0.0, 0.0));
    assert!(d.color != Vec3::new(0.0, 0.0, 0.0));
    assert!(d.ambient[0] > 0.0 && d.ambient[0] < 1.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib renderer::light`
Expected: FAIL — module/types don't exist yet.

- [ ] **Step 3: Implement `light.rs`**

Define `DirectionalLight` (fields + `Copy/Clone/Debug`) and its `Default` (pick concrete sun values: a downward-forward-right travel direction, white-ish `color`, and ambient ≈ `0.1`; record the exact values in the spec Outcome). Define `LightUniforms` with the padding fields and `From<&DirectionalLight>` (fields copied, pads `0.0`). Define `const LIGHT_BINDING: u32 = 0;` and `Light` wrapping `PerFrameUniform::new(device, allocator, size_of::<LightUniforms>() as u64, LIGHT_BINDING, vk::ShaderStageFlags::FRAGMENT)`, with `update` forwarding `bytemuck::bytes_of(&LightUniforms::from(light))` and the accessors/`destroy` forwarding to the inner helper. Add `pub mod light;` to `src/renderer/mod.rs`.

- [ ] **Step 4: Run to verify the light tests pass**

Run: `cargo test --lib renderer::light`
Expected: PASS.

- [ ] **Step 5: Widen the scene pipeline to three set layouts (caller side)**

`GraphicsPipeline::new`/`create` already accept `set_layouts: &[vk::DescriptorSetLayout]` (spec 010). No signature change — only the callers in `mod.rs` (Step 7) pass a 3-element slice. Confirm by reading `pipeline.rs` that the slice is forwarded to `PipelineLayoutCreateInfo::set_layouts` unchanged.

- [ ] **Step 6: Thread the light through `draw_frame`/`record` and bind set 2**

In `src/renderer/command.rs`:
- Add `use super::light::{DirectionalLight, Light};`.
- `draw_frame` gains params `light: &Light, directional_light: &DirectionalLight`, placed immediately after the existing `camera: &CameraUniforms` param (the two grouped together). In the uniform-update step next to `uniforms.update(frame, camera);`, add `light.update(frame, directional_light);`.
- `record` gains a `light_set: vk::DescriptorSet` param. Immediately after the existing camera `cmd_bind_descriptor_sets(..., 0, &[descriptor_set], &[])`, add the set-2 bind:
  ```rust
  dev.cmd_bind_descriptor_sets(
      cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline_layout, 2, &[light_set], &[],
  );
  ```
- At the `self.record(...)` call, pass `light.descriptor_set(frame)` for `light_set`.
- Keep the `#[allow(clippy::too_many_arguments)]` on `draw_frame` (and add it to `record` if clippy flags it).

The existing `emit_object_draws` tests are unchanged and must still pass (empty slice, mixed-order) — the light bind lives in `record`, outside the tested helper.

- [ ] **Step 7: Own the light in the renderer and add `set_directional_light`**

In `src/renderer/mod.rs`:
- Add `use light::{DirectionalLight, Light};` and fields `pub(crate) light: light::Light,` and `pub(crate) directional_light: DirectionalLight,`.
- In `new`, in the fallible `resources` closure, after `uniforms` and `texture_set_layout`/`white_texture` are built, create `let light = light::Light::new(&device, &mut allocator)?;` held in an `Option` for rollback (destroy it in the error arm before the texture/layout cleanup). Change the `GraphicsPipeline::new` call's layout slice to `&[uniforms.as_ref().unwrap().descriptor_set_layout(), layout, light.as_ref().unwrap().descriptor_set_layout()]`. Set `directional_light: DirectionalLight::default()` and move `light` into the struct.
- In the resize path, change the rebuild's layout slice to `&[self.uniforms.descriptor_set_layout(), self.texture_set_layout, self.light.descriptor_set_layout()]`.
- In `render_objects_with_text`, pass `&self.light, &self.directional_light` into `draw_frame` immediately after the `&camera` argument (matching the Step 6 param order).
- Add the method:
  ```rust
  /// Set the directional light used for subsequent frames. Takes effect on the next render call.
  pub fn set_directional_light(&mut self, light: DirectionalLight) {
      self.directional_light = light;
  }
  ```
- In `Drop`, next to `self.uniforms.destroy(&self.device, &mut self.allocator);`, add `self.light.destroy(&self.device, &mut self.allocator);` (before the allocator is dropped).
- In `src/lib.rs`, add `pub use renderer::light::DirectionalLight;`.

- [ ] **Step 8: Update, recompile, and validate the shaders**

`shaders/triangle.vert`: add `layout(location = 3) in vec3 inNormal;` and `layout(location = 2) out vec3 fragNormal;` (keep `fragColor = 0`, `fragUV = 1`); in `main`, `mat3 n = transpose(inverse(mat3(object.model))); fragNormal = normalize(n * inNormal);`.

`shaders/triangle.frag`: add `layout(location = 2) in vec3 fragNormal;`, the set-2 `LightUniforms` block (`direction`/`color`/`ambient`), and:
```glsl
vec3 N = normalize(fragNormal);
vec3 L = normalize(-light.direction);
float diff = max(dot(N, L), 0.0);
vec3 shade = light.ambient + light.color * diff;
vec4 albedo = texture(tex, fragUV) * vec4(fragColor, 1.0);
outColor = vec4(shade, 1.0) * albedo;
```

Recompile and validate:
```bash
glslc shaders/triangle.vert -o shaders/triangle.vert.spv
glslc shaders/triangle.frag -o shaders/triangle.frag.spv
spirv-val shaders/triangle.vert.spv
spirv-val shaders/triangle.frag.spv
```

- [ ] **Step 9: Verify all gates pass**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean; `renderer::pipeline::tests::*_spirv_parses_and_starts_with_magic`, the light-layout tests, and the unchanged `emit_object_draws` tests all pass.

- [ ] **Step 10: GPU verification — lighting is applied, no validation errors**

Run (outside the sandbox, on a machine with the Vulkan driver):
```bash
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo run --example cube
```
Expected: the cubes are now shaded by the default sun — faces toward the light brighter, faces away dim (not black, thanks to ambient); textures still show, modulated by light; F3 overlay still toggles; resize/minimize/restore/close produce zero validation (incl. synchronization) errors; gpu-allocator clean on exit. (Shading looks smooth here because the cube still has Task 1's placeholder normals; Task 4 makes it flat per-face.)

- [ ] **Step 11: Commit**

```bash
git add src/renderer/light.rs src/renderer/pipeline.rs src/renderer/command.rs \
        src/renderer/mod.rs src/lib.rs \
        shaders/triangle.vert shaders/triangle.vert.spv shaders/triangle.frag shaders/triangle.frag.spv
git commit -m "feat(renderer): directional light with diffuse and ambient shading"
```

---

### Task 4: Cube demo — per-face normals, free-fly camera, reset

Example-only. Makes the lighting read as crisp per-face shading and lets you move around the scene to see it from every angle. No library changes.

**Files:**
- Modify: `examples/cube.rs`
- Modify: `specs/011-directional-lighting.md` (Outcome), `ROADMAP.md` (lighting note)

**Interfaces:**
- Consumes: `Vertex` (with `normal`), `Object`, `DirectionalLight` + `set_directional_light`, `render_objects_with_text`, existing frame-interval from the diagnostics loop.
- Produces: nothing library-facing.

- [ ] **Step 1: Rebuild the cube with per-face normals (24 vertices)**

Replace the 8-vertex array + 36-index array with **24 vertices** (4 per face), each face's four vertices carrying that face's outward normal (`±X`, `±Y`, `±Z`), and 36 indices (two triangles per face). Keep per-vertex colors/UVs as before (reuse the existing corner colors per face; UVs map each face to the unit square). This is consumer geometry — keep it in the example.

- [ ] **Step 2: Add a free-fly camera driven by held keys**

Track camera `position: Vec3`, `yaw: f32`, `pitch: f32`, and a set of currently-held keys (update on `WindowEvent::KeyboardInput` press/release). Each frame, using the frame delta already computed for the diagnostics overlay: WASD translate along the yaw/pitch-derived forward/right vectors; arrow keys adjust yaw (left/right) and pitch (up/down), pitch clamped to ±89°; build `view = look_at(&position, &(position + forward), &up)`. Pass the live `position` to `diagnostics.format(..., position)` (already the overlay's camera-position input) and the `view` to the render call. Store the initial pose as constants.

- [ ] **Step 3: Add the reset key and (optionally) pose the sun**

On `R` pressed, restore `position`/`yaw`/`pitch` to the initial pose. Optionally call `renderer.set_directional_light(DirectionalLight { .. })` once before the loop to pose the sun relative to the scene (otherwise the default sun is used). Import `DirectionalLight` from the crate root if used.

- [ ] **Step 4: Verify gates**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 5: GPU acceptance run**

Run (outside the sandbox):
```bash
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo run --example cube
```
Expected: each cube face is uniformly shaded by its own normal (crisp flat per-face lighting — the side toward the sun bright, the opposite side dim); WASD/arrows fly and look, **R** resets, the overlay `Camera: (x, y, z)` tracks the camera; textures still modulate; depth, resize, minimize/restore, and shutdown are clean with zero validation errors.

- [ ] **Step 6: Record the outcome**

In `specs/011-directional-lighting.md`, fill the Outcome section (what was implemented, the exact default sun values, test count, gates, the GPU acceptance result). In `ROADMAP.md`, note on the "Basic lighting" item that directional + diffuse + ambient + normals are done and specular + point/spot remain deferred (leave the box unchecked, or split it into sub-items and check only those).

- [ ] **Step 7: Commit**

```bash
git add examples/cube.rs specs/011-directional-lighting.md ROADMAP.md
git commit -m "feat(examples): per-face lit cube with free-fly camera and reset"
```
