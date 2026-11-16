# GPU Instancing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render consecutive objects that share one resolved mesh and resolved texture with a single
indexed instanced draw, replacing the current one-draw-per-object push-constant path, with no change
to the public `Object`/`render_objects` API.

**Architecture:** Per-object model + specular material data moves from a push constant (rewritten
every draw) to a per-frame-in-flight mapped instance vertex buffer (written once, in input order,
before recording). The scene pipeline gains a second vertex binding (`inputRate = INSTANCE`) read at
locations 4-8; the existing mesh vertex binding (locations 0-3) is unchanged. `command.rs`'s draw
emission walks objects in order, extending a run only while the next object resolves to the same
mesh+texture key, and emits one `vkCmdDrawIndexed(indexCount, runLength, 0, 0, firstInstance)` per
run instead of one per object. This plan deliberately separates "wire up the new instance-buffer
mechanism, one object at a time" (Task 3) from "merge consecutive runs into real multi-instance
batches" (Task 4) — the two are logically separable and the first is already a full, visually
behavior-preserving replacement of the push-constant path, worth its own review gate before the
batching logic that depends on it.

**Tech Stack:** Rust, `ash` (Vulkan 1.3 dynamic rendering), `bytemuck`, `m2s2-math`, GLSL
(`glslc`/`spirv-val`), spec 019's `HeadlessRenderer`/`assert_matches_golden` for GPU acceptance.

**Spec:** [specs/016-gpu-instancing.md](../../../specs/016-gpu-instancing.md)

## Global Constraints

- Instance payload is 80 bytes: four `mat4` rows at offsets 0/16/32/48 (each a `vec4`), then
  `vec4(specularColor.xyz, shininess)` at offset 64. The vertex shader reconstructs
  `mat4 model = transpose(mat4(row0, row1, row2, row3));` (m2s2-math stores matrices row-major, same
  convention `CameraUniforms`/`ModelPushConstants` already use).
- New instance vertex attributes: locations 4-7 are the four model rows (`R32G32B32A32_SFLOAT`,
  binding 1, `inputRate = INSTANCE`), location 8 is the specular vec4 (same binding/rate/format). The
  existing mesh attributes (locations 0-3, binding 0, `inputRate = VERTEX`) are unchanged.
- `pipeline.rs`'s `PipelineDescription.binding: vk::VertexInputBindingDescription` (singular) becomes
  `bindings: &'a [vk::VertexInputBindingDescription]` (plural) — the one shared `GraphicsPipeline::create`
  both the scene pipeline (`GraphicsPipeline::new`) and the text pipeline (`text.rs`) go through
  currently wraps the single binding as `[description.binding]`; both callers must be updated in the
  same commit as this shape change (the text pipeline keeps exactly one binding, wrapped in a
  one-element slice; only the scene pipeline eventually adds a second entry — see Task 3).
- `ModelPushConstants` (currently `model: Mat4, specular_color: Vec3, shininess: f32` at offsets
  0/64/76, visible to both shader stages) and its matching `MaterialPushConstants` push-constant view
  in `triangle.frag` are removed entirely once the instance mechanism lands (Task 3) — no parallel
  push-constant and instance-buffer paths coexist past that task.
- The per-frame instance buffer mirrors the existing `Camera`/`Light` per-frame-mapped pattern: one
  `Buffer::new_host_visible_mapped` (usage `VERTEX_BUFFER`) per frame-in-flight slot, written only
  after that slot's fence signals (same timing as the existing camera/light `update` calls in
  `command.rs`'s `draw_frame`), sized for `MAX_INSTANCES_PER_FRAME = 4096` objects. `HeadlessRenderer`
  reuses the identical type and always writes slot 0, exactly how it already reuses `Uniforms`/`Light`.
- Batching is contiguous-only: walking objects in input order, a run extends only while the next
  object's resolved `(MeshId, resolved texture identity)` key matches the current run's key. A key
  change — including a texture-only or mesh-only change — always starts a new run. A/B/A is always
  three runs/three draws, never merged. Material/model differences never split a run.
  `instanceCount = 1` is a valid, ordinary case (a run of length 1).
- Validate the whole frame (every object's mesh/texture resolution, same as spec 015's existing
  pass, plus the new instance-count cap) before acquire/mapped-writes/fence-reset — an empty
  `objects` slice still clears/presents; input exceeding `MAX_INSTANCES_PER_FRAME = 4096` returns
  `ValidationError` before any GPU work, never truncates or partially draws.
- `RendererStats` (and the renderer's internal equivalent fields) gains `scene_instances: usize`
  (total objects submitted across all runs in the frame). `scene_draw_calls` changes meaning from
  "one per object" to "one per run" (batch count). `submitted_indices` becomes the sum, over every
  submitted run, of that run's mesh's `indexCount * instanceCount`. Empty frames report zero for all
  three; a failed acquire retains the prior frame's counts; text draws never count as scene work.
- Quality gates zero-warning at every commit, whole crate: `cargo fmt --all -- --check`, `cargo
  clippy --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test
  --all-targets`. Shader changes additionally require `glslc`/`spirv-val` clean on both recompiled
  `.spv` files. Every task's own commit must pass the full suite — never split a signature/shape
  change from its call-site fixes across task boundaries (this project's pre-commit hook enforces
  the whole crate; specs 015/018 both hit this the hard way — see those plans' ledgers).
  GPU-dependent tests stay `#[ignore]`d by default, per spec 019's established pattern; this plan's
  executor has no GPU and cannot run them. Golden images are generated once by the user via
  `M2S2_UPDATE_GOLDEN=1`, same handoff specs 015/018/019 established.
- Conventional Commits; no `Co-Authored-By`/AI-attribution trailer.

## Review Focus

- **A run of exactly `MAX_INSTANCES_PER_FRAME` (4096) objects, and a frame of 4097** — the former
  must succeed, the latter must return `ValidationError` before any GPU work, with no partial draw of
  the first 4096. *Test in Task 4.*
- **Three objects sharing one mesh/texture, where the middle one's `model` is non-uniformly scaled
  along one axis** — the inverse-transpose normal matrix must still be computed per-instance in the
  shader (from that instance's own reconstructed `model`, not a batch-shared value), so lighting on
  the scaled instance must differ correctly from its unscaled neighbors in the same draw call. A test
  that only checks "the draw count went down" would miss a regression that broke per-instance normal
  correctness while still reducing draw calls. *GPU-only (shader correctness); reviewed by hand-trace
  in Task 3, golden-image-tested in Task 5.*
- **A/B/A where A and B share the same mesh but different textures (not different meshes)** — texture
  identity alone must be enough to split a run; a regression that only keyed on `MeshId` would wrongly
  merge A and B into one run and render the wrong texture on some instances. *Test in Task 4.*
- **The instance buffer slot currently being read by a submitted-but-not-yet-fenced frame** — writing
  the next frame's instance data must wait for that slot's fence, exactly like the existing
  camera/light update timing; a test can't directly observe GPU timing, but the write call site must
  be placed identically to the existing camera/light `update` calls, not earlier. *Reviewed by
  inspection in Task 3 (placement) and Task 5 (GPU resize/teardown scenario).*
- **An object whose resolved texture is `None` (the white default) immediately followed by an object
  with the same mesh and an explicit texture** — `None` and `Some(white's own TextureId)` are
  different resolved identities (spec 015's `resolve_texture_set` returns a concrete
  `vk::DescriptorSet` either way, but the two should not be assumed equal just because both end up
  binding the same physical white descriptor set by coincidence) — confirm the batch key uses
  `Option<TextureId>` (or the resolved set itself) consistently, not a semantic "is it white"
  check that could accidentally merge an explicit-white object with a textureless one in a way the
  spec didn't actually ask for (merging them is fine if the key is the resolved descriptor set itself,
  since both would legitimately bind white — this is a reviewer judgment call to make explicit, not
  a bug to assume). *Test in Task 4.*

---

### Task 1: Instance data type and per-frame instance buffer

**Files:**
- Modify: `src/renderer/pipeline.rs` (new `InstanceData` type, alongside the still-present
  `ModelPushConstants` — not yet removed)
- Create: `src/renderer/instance_buffer.rs` (the new per-frame-mapped instance buffer owner)

**Interfaces:**
- Consumes: nothing from other tasks — this is the plan's foundation, pure addition.
- Produces: `#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)] pub(crate) struct InstanceData { pub
  model_row0: Vec4, pub model_row1: Vec4, pub model_row2: Vec4, pub model_row3: Vec4, pub specular:
  Vec4 }` (80 bytes, offsets 0/16/32/48/64 — note `Vec4` four times, not a single `Mat4` field, since
  the GLSL side reads four separate `vec4` attribute locations, not one matrix attribute); a
  constructor or `From<(Mat4, Vec3, f32)>`-style conversion building the four rows from a row-major
  `Mat4` plus specular color/shininess; `pub(crate) const MAX_INSTANCES_PER_FRAME: usize = 4096;
  pub(crate) struct InstanceBuffer { ... }` with `new(device, allocator, slots: usize) -> Result<Self>`
  (sized for `MAX_INSTANCES_PER_FRAME` instances per slot), `write(&mut self, frame: usize, data:
  &[InstanceData]) -> Result<()>`, `fn buffer(&self, frame: usize) -> vk::Buffer`, `destroy(self,
  device, allocator)` — mirror `Buffer::new_host_visible_mapped`/`write_mapped`'s existing pattern and
  `Light`'s/`Uniforms`' existing per-frame-array-of-`Buffer` shape exactly (an array/Vec of `Buffer`,
  not a new lower-level abstraction).

- [ ] **Step 1: Write the failing test for `InstanceData`'s layout**

```rust
#[test]
fn instance_data_matches_shader_layout() {
    assert_eq!(std::mem::size_of::<InstanceData>(), 80);
    assert_eq!(std::mem::offset_of!(InstanceData, model_row0), 0);
    assert_eq!(std::mem::offset_of!(InstanceData, model_row1), 16);
    assert_eq!(std::mem::offset_of!(InstanceData, model_row2), 32);
    assert_eq!(std::mem::offset_of!(InstanceData, model_row3), 48);
    assert_eq!(std::mem::offset_of!(InstanceData, specular), 64);
}
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `cargo test --lib renderer::pipeline`
Expected: FAIL — `InstanceData` not defined.

- [ ] **Step 3: Implement `InstanceData` and its conversion from `(Mat4, Vec3, f32)`**

Extract each row of a row-major `Mat4` as a `Vec4` (row `i`'s four components), matching how
`m2s2-math` already exposes row access (check `m2s2-math`'s `Mat4` API for the idiomatic way to get
one row as a `Vec4` — likely an index or a `row(i)` method; use whatever's already idiomatic rather
than manually indexing scalars).

- [ ] **Step 4: Run test, confirm it passes**

Run: `cargo test --lib renderer::pipeline`
Expected: PASS.

- [ ] **Step 5: Write failing tests for `InstanceBuffer`**

Cover: `new` with `slots = 2` produces two independently-writable buffers; `write(0, data)` then
`write(1, other_data)` don't clobber each other (read back via the existing test-only
`Buffer::mapped_bytes`/equivalent accessor, if one is reachable — if not, test only that `write`
returns `Ok` for a valid-length slice and `Err` for a slice longer than `MAX_INSTANCES_PER_FRAME`,
which is fully CPU-testable without GPU readback).

- [ ] **Step 6: Run tests, confirm they fail**

Run: `cargo test --lib renderer::instance_buffer`
Expected: FAIL — type not defined.

- [ ] **Step 7: Implement `InstanceBuffer`**

GPU-dependent construction (`Buffer::new_host_visible_mapped` calls) is not independently
unit-testable without a driver — write it, get it to compile, structurally mirror how `Light`/
`Uniforms` already construct their own per-frame `Buffer` arrays in `mod.rs`'s `init`.

- [ ] **Step 8: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean. Nothing outside this task's own tests calls `InstanceData`/`InstanceBuffer` yet
— if clippy flags anything as dead code, gate it with a temporary, clearly-commented
`#[allow(dead_code)]` naming Task 3 as the first real caller (same established pattern as specs
015/018's own Task 1s).

- [ ] **Step 9: Commit**

```bash
git add src/renderer/pipeline.rs src/renderer/instance_buffer.rs
git commit -m "feat(renderer): instance data layout and per-frame instance buffer"
```

---

### Task 2: `PipelineDescription`'s multi-binding shape (zero behavior change)

**Files:**
- Modify: `src/renderer/pipeline.rs` (`PipelineDescription`, `GraphicsPipeline::create`'s vertex
  input construction, `GraphicsPipeline::new`'s own `PipelineDescription` literal)
- Modify: `src/renderer/text.rs` (its `PipelineDescription` literal)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `PipelineDescription.bindings: &'a [vk::VertexInputBindingDescription]` (plural,
  replacing today's singular `binding` field) — this is purely a shape change with NO behavior
  change in this task: `GraphicsPipeline::new` wraps `Vertex::binding_description()` in a one-element
  slice (unchanged pipeline behavior), `text.rs` wraps its existing binding the same way. Task 3
  consumes this by appending a second binding entry for the new instance buffer.

- [ ] **Step 1: Change `PipelineDescription.binding` to `bindings`**

In `pipeline.rs`, rename the field and change its type from `vk::VertexInputBindingDescription` to
`&'a [vk::VertexInputBindingDescription]`. In `Self::create`, replace `let binding_descriptions =
[description.binding];` with `let binding_descriptions = description.bindings;` (the array-wrap is no
longer needed — the field is already a slice).

- [ ] **Step 2: Update both callers**

`GraphicsPipeline::new`'s own `PipelineDescription` literal: `binding: Vertex::binding_description()`
becomes `bindings: &[Vertex::binding_description()]`. `text.rs`'s `PipelineDescription` literal:
wrap its existing binding expression in a one-element slice literal the same way.

- [ ] **Step 3: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean, identical pass/ignored counts to the branch baseline — this task changes no
runtime behavior anywhere, only the shape of how a binding list is passed internally.

- [ ] **Step 4: Commit**

```bash
git add src/renderer/pipeline.rs src/renderer/text.rs
git commit -m "refactor(renderer): PipelineDescription takes a binding slice instead of one binding"
```

---

### Task 3: Shaders, the real instance binding, and one-draw-per-object via the new mechanism

**This is the highest-risk task in this plan** — it touches GLSL shader source, the pipeline's
vertex input layout, and `command.rs`'s draw emission together, and none of it is verifiable without
a GPU. It deliberately does NOT yet implement real multi-object batching (that's Task 4) — every
object still gets its own `vkCmdDrawIndexed(..., instanceCount=1, ..., firstInstance=N)` call, so the
*visual* and *diagnostic* output (`scene_draw_calls == objects.len()`, same as before this task)
should be unchanged from the pre-016 baseline. This task's entire purpose is proving the new
mechanism (instance buffer → vertex attributes → matrix reconstruction → fragment shading) produces
pixel-identical output to the old push-constant mechanism, in isolation from the batching logic that
depends on it being correct.

**Files:**
- Modify: `shaders/triangle.vert`, `shaders/triangle.frag` (and their committed `.spv` binaries)
- Modify: `src/renderer/pipeline.rs` (`GraphicsPipeline::new`'s real two-binding vertex input,
  instance attribute descriptions, `ModelPushConstants`/`model_push_range`/`validate_push_limit`
  removed entirely)
- Modify: `src/renderer/command.rs` (write every object's `InstanceData` into the per-frame instance
  buffer in input order; `ObjectCommand`/`emit_object_draws` lose `PushModel` and gain a per-object
  `DrawIndexedInstanced`-style single-instance draw using that object's own `firstInstance` offset)
- Modify: `src/renderer/mod.rs`, `src/renderer/headless.rs` (construct the new `InstanceBuffer`
  alongside the existing `Uniforms`/`Light` construction in `init`; both renderers' Drop/teardown
  gain the new owner, mirroring exactly how `Light`/`Uniforms` are already torn down)

**Interfaces:**
- Consumes: Task 1's `InstanceData`/`InstanceBuffer`/`MAX_INSTANCES_PER_FRAME`; Task 2's
  `PipelineDescription.bindings` shape.
- Produces: the scene pipeline's real two-binding vertex input (mesh binding 0 unchanged, new
  instance binding 1 at locations 4-8); `ModelPushConstants` and `MaterialPushConstants` fully gone
  from both the Rust side and both shaders; `command.rs`'s draw emission writes instance data once
  per frame and draws with `firstInstance`, consumed by Task 4's real batching logic.

- [ ] **Step 1: Rewrite `shaders/triangle.vert`**

Remove the `layout(push_constant, row_major) uniform ModelPushConstants { mat4 model; }` block. Add
instance attributes at locations 4-7 (`vec4 inModelRow0/1/2/3`) and location 8 (`vec4 inSpecular`).
Reconstruct `mat4 model = transpose(mat4(inModelRow0, inModelRow1, inModelRow2, inModelRow3));` and
use it exactly where `object.model` was used before (world position, normal matrix). Add `layout
(location = 4) flat out vec4 fragSpecular;` (pick an unused output location — check which locations
0-3 already use as outputs before choosing 4, since the existing fragment-input locations in
`triangle.frag` are separate from these new vertex-input attribute locations but must not collide
with this shader's own existing *output* locations) and set it to `inSpecular` unchanged.

- [ ] **Step 2: Rewrite `shaders/triangle.frag`**

Remove the `layout(push_constant) uniform MaterialPushConstants { ... } material;` block and its
redeclared-subset comment. Add `layout(location = 4) flat in vec4 fragSpecular;` matching Step 1's
output location. Replace every `material.specularColor`/`material.shininess` read with
`fragSpecular.rgb`/`fragSpecular.a`. No other change to this shader's shading math (the directional
and point-light loops, `blinnPhongSpecular`, ambient — all unchanged).

- [ ] **Step 3: Recompile and validate both shaders**

Run: `glslc shaders/triangle.vert -o shaders/triangle.vert.spv && glslc shaders/triangle.frag -o
shaders/triangle.frag.spv && spirv-val shaders/triangle.vert.spv && spirv-val shaders/triangle.frag.spv`
Expected: no errors or warnings.

- [ ] **Step 4: Wire the real instance binding into `GraphicsPipeline::new`**

Build the instance binding (`binding(1).stride(80).input_rate(INSTANCE)`) and its five attribute
descriptions (locations 4-8, `R32G32B32A32_SFLOAT`, offsets from `InstanceData`'s field offsets via
`std::mem::offset_of!`, matching Task 1's exact layout). Pass `bindings: &[Vertex::binding_description(),
instance_binding]` and `attributes: &[Vertex's 4, ...instance's 5].concat()` (or however you choose
to merge two attribute arrays — a plain concatenation into one combined slice/Vec before passing it
in is the simplest approach). Remove `model_push_range()`'s call from `push_ranges` (now `&[]`) and
delete `ModelPushConstants`, `MODEL_PUSH_OFFSET/SIZE/STAGES`, `model_push_range`, and
`validate_push_limit` (and its call in `GraphicsPipeline::new`) entirely — nothing needs a
push-constant range anymore.

- [ ] **Step 5: Construct `InstanceBuffer` in `mod.rs`'s `init` and `headless.rs`'s constructor**

Mirror exactly how `Light::new`/`Uniforms::new` are already constructed and torn down in both
files' rollback-on-error construction closures and `Drop` implementations — same ordering discipline,
same `Option`-wrapped field pattern for `VulkanRenderer`, same plain field for `HeadlessRenderer` if
that's how `Light`/`Uniforms` are already held there (check before assuming either shape).
`VulkanRenderer` sizes it for `FRAMES_IN_FLIGHT` slots; `HeadlessRenderer` sizes it for 1 slot and
always writes/reads slot 0, exactly like its existing `Uniforms`/`Light` usage.

- [ ] **Step 6: Rewrite `command.rs`'s draw emission for one-instance-per-object**

Before recording (same point in `draw_frame` where camera/light are currently updated, after the
fence wait for this frame slot), build a `Vec<InstanceData>` in input order from `objects`, one entry
per object (`object.model`, `object.specular.color`, `object.specular.shininess`), and write it via
`InstanceBuffer::write(frame, &instances)`. In `emit_object_draws`/`ObjectCommand`: remove
`PushModel`; add a `BindInstanceBuffer(vk::Buffer)` variant (bound once per frame, not per object,
analogous to how the mesh vertex/index buffers are now bound per-object but the instance buffer is
the SAME buffer for every object this frame — bind it once before the loop, the same way the camera/
light descriptor sets are already bound once per frame rather than per object); change
`DrawIndexed(u32)` to also carry a `first_instance: u32` (or add a new variant) and record
`cmd_draw_indexed(cmd, index_count, 1, 0, 0, first_instance)` — `instanceCount` is hardcoded to `1` in
THIS task (Task 4 makes it real).

- [ ] **Step 7: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean, full stop. Existing `command.rs` tests that asserted `PushModel` commands need
rewriting to assert the new instance-buffer-bind + `first_instance`-bearing draw instead — do this
rewrite in this same task/commit, not deferred, since the old assertions will fail to compile/pass
against the new `ObjectCommand` shape.

- [ ] **Step 8: Commit**

```bash
git add shaders/triangle.vert shaders/triangle.frag shaders/triangle.vert.spv shaders/triangle.frag.spv src/renderer/pipeline.rs src/renderer/command.rs src/renderer/mod.rs src/renderer/headless.rs
git commit -m "feat(renderer): per-object instance buffer replaces the model/material push constant"
```

---

### Task 4: Real contiguous batching

**Files:**
- Modify: `src/renderer/command.rs` (the batching walk; `ObjectCommand`'s draw variant gains a real
  `instance_count`)
- Modify: `src/renderer/mod.rs` (`scene_draw_calls`/`submitted_indices`/new `scene_instances`
  computation; the `MAX_INSTANCES_PER_FRAME` capacity check in the existing validation pass)
- Modify: `src/renderer/diagnostics.rs` (`RendererStats` gains `scene_instances: usize`, formatted
  alongside the existing counters)

**Interfaces:**
- Consumes: Task 3's per-object instance-buffer mechanism (this task changes HOW MANY objects one
  draw call covers, not how instance data gets to the shader).
- Produces: `emit_object_draws` (or whatever it's now named) walks objects and emits one draw per
  contiguous compatible run, not one per object; `RendererStats.scene_instances`.

- [ ] **Step 1: Write failing tests for the batching walk**

Using the actual production emission function (not a simulated/parallel test implementation — the
spec explicitly forbids a separate test-only batching model), cover: empty input (no commands); a
single object (one run of length 1); all objects sharing one mesh/texture (one run, length =
objects.len()); strictly alternating mesh/texture (N runs of length 1 each); A/A/B/B/A (three runs:
length 2, length 2, length 1, in that exact order) — assert each run's `first_instance` offset,
length, and mesh's index count.

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cargo test --lib renderer::command`
Expected: FAIL — current emission still draws one-per-object.

- [ ] **Step 3: Implement the batching walk**

Walk `objects` in order, resolving each one's mesh/texture (reusing Task 3/spec-015's existing
resolution — no second implementation). Extend the current run while the resolved `(MeshId,
resolved-texture-identity)` key matches; otherwise close the current run and start a new one. Emit
one bind-mesh + bind-texture + (if Task 3's instance buffer isn't already bound once per frame,
confirm it still only binds once) + `cmd_draw_indexed(index_count, run_length, 0, 0,
run.first_instance)` per run.

- [ ] **Step 4: Run tests, confirm they pass**

Run: `cargo test --lib renderer::command`
Expected: PASS.

- [ ] **Step 5: Write failing tests for the `MAX_INSTANCES_PER_FRAME` capacity check**

Exactly 4096 objects accepted; 4097 rejected with `ValidationError` before any mutation (reuse the
existing early-validation-pass seam from spec 015 — add the length check there, same place as the
mesh/texture resolution checks, before `pending_resize`/acquire).

- [ ] **Step 6: Run test, confirm it fails; implement; confirm it passes**

Run: `cargo test --lib renderer::mod` (or wherever the validation pass's own tests live)
Expected: FAIL, then implement the check, then PASS.

- [ ] **Step 7: Update `scene_draw_calls`/`submitted_indices`/`scene_instances`**

`scene_draw_calls` becomes the number of runs (not `objects.len()`). `submitted_indices` becomes the
checked sum, over every run, of `run.mesh.index_count as u64 * run.length as u64`. `scene_instances`
is the checked sum of every run's length (equivalently just `objects.len()` when the frame
succeeds, but compute it the same way the other two counters are computed — from the actual runs —
for consistency and to make a future per-run skip/reject path trivial to extend correctly). Add
`scene_instances` to `RendererStats` and its `Display`/diagnostics-formatting counterpart.

- [ ] **Step 8: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 9: Commit**

```bash
git add src/renderer/command.rs src/renderer/mod.rs src/renderer/diagnostics.rs
git commit -m "feat(renderer): merge consecutive compatible objects into real instanced draws"
```

---

### Task 5: GPU golden-image acceptance tests, user handoff

**Files:**
- Modify: `src/renderer/headless.rs`

**Interfaces:**
- Consumes: Task 4's finished batching (every scenario below exercises the complete, real
  multi-instance behavior).
- Produces: nothing for later tasks — this plan's final task.

- [ ] **Step 1: Write the new golden-image scenario tests**

Following the existing `self_test_scene_matches_golden_with_zero_validation_errors`/spec 015's
multi-mesh test structure (fixed-camera helper, `assert_matches_golden` at tolerance 2,
`renderer.validation_error_count() == 0`), add `#[ignore]`d tests for:
- Many instances (e.g. 20+) of one mesh/texture with independently different transforms and
  shininess values, including at least one non-uniformly-scaled instance (per this plan's Review
  Focus) — assert `scene_draw_calls == 1` (not 20) alongside the golden-image comparison, proving
  the draw-count reduction this spec exists for, not just a visual match.
- Several mesh/texture runs in one frame (e.g. mesh A ×3, mesh B ×2, mesh A ×1 again) — assert
  `scene_draw_calls == 3` matching the three expected runs.
- Non-uniform scale lighting correctness in isolation (could be folded into the first scenario if the
  golden image alone demonstrates it clearly — your judgment call on whether a dedicated scenario
  adds real signal beyond the first one).
- Text drawn alongside an instanced scene (reuse `render_frame_with_text`, already built in spec
  015) — confirms the text pass still records correctly after the batched scene draws, per this
  spec's own test-plan requirement to validate text after the instanced scene specifically, not just
  text-only (already covered by spec 015's `multi_mesh_text_only` scenario).
Each needs its own distinct `#[ignore]` reason string naming its own golden PNG path under
`testdata/golden/` (per spec 018's ledgered convention — never reuse an identical reason string
across two different tests).

- [ ] **Step 2: Build and run the full gate suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean; every new test compiles; GPU-dependent tests remain `#[ignore]`d.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/headless.rs
git commit -m "test(renderer): golden-image GPU acceptance tests for instanced batching"
```

- [ ] **Step 4: Hand off to the user — generate the new golden images**

This plan's executor has no GPU and cannot complete this step. Report clearly that the user must
run, on a machine with a Vulkan driver, one `M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless
-- --ignored <test_name>` invocation per new scenario from Step 1 to generate each golden PNG, then
run the same set of tests **without** `M2S2_UPDATE_GOLDEN` to confirm they pass, and finally re-run
the full pre-existing `#[ignore]`d suite (every test from specs 015/018/019) to confirm nothing
regressed — this spec touches the shared scene pipeline/shaders, so a regression risk extends to
every prior GPU-tested scene, not just this spec's own new tests. Only after the user confirms this
should the golden PNGs be committed and spec 016's Outcome section + `ROADMAP.md` be updated to
reflect verified (not just implemented) status — the same two-step handoff pattern specs 015/018/019
established.
