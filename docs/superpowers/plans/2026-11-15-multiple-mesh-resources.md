# Multiple Mesh Resources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `VulkanRenderer` upload several independent indexed meshes after construction and
reference them per-object through a typed, renderer-checked `MeshId`, instead of baking one mesh in
at construction time.

**Architecture:** A new `handle.rs` module gives every `VulkanRenderer` instance a distinct
process-wide owner id; `MeshId` and the strengthened `TextureId` both carry `(owner, slot)` and are
only resolvable against the renderer that produced them. A new `mesh.rs` owns `StoredMesh` (one
device-local vertex buffer + index buffer + index count) and an append-only mesh store, mirroring
`texture.rs`'s existing `StoredTexture`/texture-store shape. `Object` relocates to a new `scene.rs`
and gains a `mesh: MeshId` field. The constructors drop their `vertices`/`indices` parameters in
favor of a new `create_mesh`. Handle validation moves earlier: `render_objects_with_text` resolves
every object's `MeshId`/`TextureId` against the renderer's stores in one pass before any resize/
acquire/fence work — mirroring its existing camera/specular validation — so an invalid handle fails
the whole frame up front instead of being discovered mid-recording. `command.rs`'s per-object draw
emission moves the vertex/index buffer binds from a single per-frame call into the per-object path,
resolved by each object's own mesh.

**Tech Stack:** Rust, `ash` (Vulkan 1.3 dynamic rendering), `bytemuck`, `m2s2-math`, spec 019's
`HeadlessRenderer`/`assert_matches_golden` for GPU acceptance.

**Spec:** [specs/015-multiple-mesh-resources.md](../../../specs/015-multiple-mesh-resources.md)

## Global Constraints

- `MeshId`/`TextureId` both carry `(owner: u64, slot: usize)`. The owner id comes from a process-wide
  `AtomicU64` counter in the new `handle.rs`, allocated once per `VulkanRenderer` construction and
  stored on the renderer. Resolving either handle type checks `owner` equality before indexing by
  `slot` — a handle from one renderer instance must never resolve against another's store even if
  the slot numbers collide. There is exactly one identity-check implementation, shared by both
  handle types, not two.
- **`TextureId`'s behavior changes**: an out-of-range or foreign `Some(TextureId)` now returns
  `ValidationError` instead of silently falling back to white. `None` remains the only way to
  request the white default. The existing `resolve_none_and_out_of_range_fall_back_to_white` test in
  `texture.rs` is rewritten to assert rejection, not fallback.
- **Validation happens in a separate pass, before any per-frame GPU work** — not inline during
  command recording. `render_objects_with_text` validates every object's `MeshId`/`TextureId`
  against the renderer's stores in the same pass as its existing camera/specular validation, before
  checking `pending_resize` or touching acquire/fence/mapped-write state. An invalid handle fails the
  whole frame with `ValidationError`; no object in that call is partially submitted. By the time
  `FrameContext::record`/`emit_object_draws` runs, every object's handles are already known-valid.
- `StoredMesh` owns one device-local vertex buffer, one index buffer, and its index count, built via
  the existing `Buffer::new_device_local` helper (the same one `buffer.rs` already uses today). If
  the vertex buffer succeeds and the index buffer upload then fails, destroy the vertex buffer before
  returning the error — mirroring `StoredTexture::new`'s existing sibling-resource rollback. A failed
  upload leaves every existing handle valid and publishes no partial store entry.
- Mesh upload validation (before any GPU allocation): empty vertices or indices rejected; any index
  value outside the supplied vertex slice rejected; an index count not divisible by three rejected;
  checked (non-panicking) byte/count-conversion overflow. Existing finite/invertible-transform and
  normal-input requirements are unchanged.
- `command.rs`'s `cmd_bind_vertex_buffers`/`cmd_bind_index_buffer` move from their current single
  per-frame call (outside the object loop) into the per-object path in `ObjectCommand`/
  `emit_object_draws`, resolved per object instead of supplied once as an external `index_count`.
  The existing `mixed_objects_bind_resolved_set_then_push_then_draw_in_order` test and its
  `record_objects` helper (both currently hardcode one shared index count for every object) are
  rewritten to exercise interleaved different-mesh objects (mesh A/B/A).
- Both constructors (`with_window`, `unsafe new`) lose their `vertices`/`indices` parameters.
  `render_frame` gains a leading `mesh: MeshId` parameter. `examples/triangle.rs` and
  `examples/cube.rs` migrate in the same change; `examples/window.rs` never constructs a
  `VulkanRenderer` and needs no change.
- `submitted_indices` becomes the checked sum of each submitted object's own mesh index count
  (previously `objects.len() as u64 * one global index count`). Empty frames report zero; a failed
  acquire retains the prior submitted count. Text does not count as scene work.
- Quality gates zero-warning at every commit: `cargo fmt --all -- --check`, `cargo clippy
  --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test
  --all-targets`. Every task's own commit must pass the full suite — never split a signature change
  from its call-site fixes across task boundaries (this project's pre-commit hook enforces the whole
  crate, not per-file; a prior plan on this branch structure got this wrong and had to be corrected
  mid-flight — see the ledger of spec 018's plan for the incident).
  GPU-dependent tests stay `#[ignore]`d by default, per spec 019's established pattern; this plan's
  executor has no GPU and cannot run them. Golden images are generated once by the user via
  `M2S2_UPDATE_GOLDEN=1`, same handoff spec 019/018 established.
- Conventional Commits; no `Co-Authored-By`/AI-attribution trailer.

## Review Focus

- **A frame with one valid object and one object whose `MeshId` belongs to a *different, still-live*
  `VulkanRenderer` instance** — the whole frame must be rejected with `ValidationError` before any
  GPU work, not just a slot-range check. This is the entire reason the owner id exists; a test that
  only checks an out-of-range slot (not a foreign-but-in-range owner) wouldn't catch a regression
  that dropped the owner check and fell back to pure slot indexing. *Test in Task 1 (handle
  resolution) and Task 4 (the renderer-level validation pass).*
- **`create_mesh` called with indices that reference a vertex index exactly equal to
  `vertices.len()`** (one past the end, the classic off-by-one) — must be rejected, not accepted as
  if 0-indexed-inclusive. *Test in Task 2.*
- **A mesh upload whose vertex buffer succeeds and index buffer fails** — the vertex buffer must not
  leak, and the mesh store must gain no new entry; every previously-published `MeshId` must stay
  resolvable afterward. *Structural guarantee, reviewed by inspection in Task 2 (mirrors
  `StoredTexture::new`'s existing pattern) — not independently unit-testable without GPU fault
  injection.*
- **Two interleaved objects referencing different meshes in one `render_objects` call (mesh A, mesh
  B, mesh A again)** — each must bind its own vertex/index buffers and draw with its own index count,
  not the first object's buffers/count reused for all three. *Test in Task 4 (the rewritten
  `emit_object_draws` test).*
- **An empty `objects` slice passed to `render_objects`** — must emit no draw calls and report zero
  `submitted_indices`/`scene_draw_calls`, not panic on an empty mesh-resolution pass or divide by a
  mesh count of zero anywhere in diagnostics. *Test in Task 4 (already covered by the existing
  `empty_objects_emit_no_commands` test; confirm it still holds against the new per-object mesh-bind
  path, not just the texture-bind path it originally covered.)*

---

### Task 1: `handle.rs` — shared owner-id mechanism

**Files:**
- Create: `src/renderer/handle.rs`
- Modify: `src/renderer/mod.rs` (register the new module)

**Interfaces:**
- Produces: `pub(crate) fn next_owner_id() -> u64` (a process-wide monotonic counter, starting above
  0 so a default/zeroed value is never a valid owner id — e.g. the first call returns 1); a small
  helper or convention for "does this `(owner, slot)` pair resolve against `my_owner_id` and a store
  of length `my_store.len()`" that Tasks 2 and 3 both call, so the identity check has one
  implementation. Shape this however reads most naturally in Rust (a free function taking `owner`,
  `slot`, `my_owner_id`, `len` and returning `Option<usize>`, or a tiny struct — your choice; Task 2
  and Task 3's briefs will reference whatever you name it here, so pick one and document it clearly
  in this file's doc comments).
- Consumes: nothing from other tasks — this is the plan's foundation.

- [ ] **Step 1: Write the failing test for monotonic uniqueness**

```rust
#[test]
fn next_owner_id_never_repeats_across_many_calls() {
    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
        assert!(seen.insert(next_owner_id()), "owner id repeated");
    }
}
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `cargo test --lib renderer::handle`
Expected: FAIL — `next_owner_id` not defined.

- [ ] **Step 3: Implement `next_owner_id`**

A private `static COUNTER: AtomicU64` (starting at 0) and `next_owner_id()` returning
`COUNTER.fetch_add(1, Ordering::Relaxed) + 1` (so the first real id is 1, never 0) is the simplest
correct implementation. `Relaxed` ordering is sufficient — callers only need uniqueness, not
ordering relative to other state.

- [ ] **Step 4: Write the failing test for the shared resolution helper**

Write 2-3 tests against whatever shape you chose in Step 1's Interfaces note: a matching owner +
in-range slot resolves to `Some`; a mismatched owner (even with an in-range slot) resolves to
`None`; an out-of-range slot (even with a matching owner) resolves to `None`.

- [ ] **Step 5: Run tests, confirm they fail**

Run: `cargo test --lib renderer::handle`
Expected: FAIL — helper not defined.

- [ ] **Step 6: Implement the resolution helper**

- [ ] **Step 7: Run tests, confirm they pass**

Run: `cargo test --lib renderer::handle`
Expected: PASS.

- [ ] **Step 8: Register the module and run the full gate suite**

Add `mod handle;` to `src/renderer/mod.rs`'s module list (alphabetically placed, `pub(crate)` —
nothing in this module is public API). Run: `cargo fmt --all -- --check && cargo clippy
--all-targets --all-features -- -D warnings && cargo build --all-targets && cargo test
--all-targets`.
Expected: all clean, same pass/ignored counts as the branch baseline plus this task's new tests.

- [ ] **Step 9: Commit**

```bash
git add src/renderer/handle.rs src/renderer/mod.rs
git commit -m "feat(renderer): shared per-renderer owner-id mechanism for MeshId/TextureId"
```

---

### Task 2: `mesh.rs` — `MeshId`, `StoredMesh`, the mesh store

**Files:**
- Create: `src/renderer/mesh.rs`
- Test: inline `#[cfg(test)] mod tests` in the same file

**Interfaces:**
- Consumes: Task 1's `next_owner_id` is NOT called here directly (the owner id belongs to the
  renderer, not the mesh store) — but this task's store/resolve function takes an `owner: u64`
  parameter supplied by its caller (Task 3) and uses Task 1's shared resolution helper internally.
- Produces: `pub struct MeshId { owner: u64, slot: usize }` (fields private — no public
  constructor, no way for a caller to fabricate one outside `create_mesh`); `pub(crate) struct
  StoredMesh` (vertex `Buffer`, index `Buffer`, index count); a validation function (e.g.
  `pub(crate) fn validate_mesh_input(vertices: &[Vertex], indices: &[Index]) -> Result<()>`) checking
  empty input, out-of-range index values, non-multiple-of-three index count, and checked
  byte/count-conversion overflow; an upload function (e.g. `pub(crate) fn upload_mesh(device,
  allocator, vertices, indices) -> Result<StoredMesh>`) building both buffers via
  `Buffer::new_device_local`, rolling back the vertex buffer if the index buffer upload fails; a
  resolution function (e.g. `pub(crate) fn resolve_mesh(id: MeshId, owner: u64, store: &[StoredMesh])
  -> Result<&StoredMesh>`) using Task 1's shared helper, returning `ValidationError` (not a silent
  fallback — there is no sensible default mesh) on any mismatch.

- [ ] **Step 1: Write failing tests for `validate_mesh_input`**

Cover: empty vertices, empty indices, an index value equal to `vertices.len()` (one past the end —
named explicitly in this plan's Review Focus), an index count not divisible by 3, and a valid
multi-triangle mesh accepted. Do not allocate huge test arrays to exercise the overflow path —
a `usize`/`u32` conversion boundary can be tested by constructing the smallest input that would
overflow the target type's checked conversion, or by testing the checked-conversion helper function
directly if you factor one out.

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cargo test --lib renderer::mesh`
Expected: FAIL — function not defined.

- [ ] **Step 3: Implement `validate_mesh_input`**

- [ ] **Step 4: Run tests, confirm they pass**

Run: `cargo test --lib renderer::mesh`
Expected: PASS.

- [ ] **Step 5: Write failing tests for `resolve_mesh`**

Cover: matching owner + valid slot resolves to the right `StoredMesh`; mismatched owner rejected;
out-of-range slot rejected. (This duplicates Task 1's own resolution-helper tests at this layer
deliberately — Task 1 tests the shared primitive in isolation, this task tests that `mesh.rs` wires
it correctly for `MeshId` specifically.)

- [ ] **Step 6: Run tests, confirm they fail**

Run: `cargo test --lib renderer::mesh`
Expected: FAIL.

- [ ] **Step 7: Implement `StoredMesh`, the upload function, and `resolve_mesh`**

`upload_mesh` is GPU-dependent and not unit-testable without a driver — write it, get it to compile,
and structurally mirror `StoredTexture::new`'s rollback-on-partial-failure pattern (read
`texture.rs`'s `StoredTexture::new` for the exact shape: build the first resource, if the second
step fails, destroy the first before returning the error). `MeshId`'s fields stay module-private
(`owner`/`slot` have no `pub` visibility) so nothing outside this module can construct one directly.

- [ ] **Step 8: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean. This module is not yet wired into `mod.rs`/`command.rs` (that's Task 3), so
nothing calls `upload_mesh`/`resolve_mesh` outside this file's own tests yet — if clippy flags
anything as dead code, gate it the same way this project has handled the identical situation before
(a temporary, clearly-commented `#[allow(dead_code)]` naming the task that adds the first real
caller — see spec 018's Task 1 for precedent — never `--no-verify`).

- [ ] **Step 9: Commit**

```bash
git add src/renderer/mesh.rs
git commit -m "feat(renderer): MeshId, StoredMesh, and the renderer-owned mesh store"
```

---

### Task 3: Strengthen `TextureId`; relocate `Object` to `scene.rs`

**Files:**
- Modify: `src/renderer/texture.rs`
- Create: `src/renderer/scene.rs`
- Modify: `src/renderer/command.rs` (the `use` of `Object`'s new location, any `TextureId(n)`
  bare-literal test construction sites that no longer compile, and its own `Object` literal at the
  test fixture around line 639/644)
- Modify: `src/renderer/mod.rs` (`render_frame`'s internal `Object` literal — see Step 5a below;
  `MeshId`'s owner id was NOT yet threaded through this file before this task ran it once and hit a
  real blocker here, so this file's own compile-preserving patch belongs in this task, not later)
- Modify: `src/renderer/headless.rs` and `examples/cube.rs` (every existing `Object` literal — see
  Step 5a; this task now owns making the WHOLE crate compile again, not just `texture.rs`/`scene.rs`)

**Interfaces:**
- Consumes: Task 1's shared owner-id resolution helper.
- Produces: `TextureId` changes from today's bare `TextureId(pub(super) usize)` to carrying an owner
  id too (exact field shape mirrors Task 1/2's `MeshId` — private fields, no public constructor);
  `resolve_texture_set` changes from `-> vk::DescriptorSet` (silent fallback) to `-> Result<
  vk::DescriptorSet>` (reject on mismatch or out-of-range `Some`, `None` still resolves to white
  infallibly); `pub struct Object` now lives in `scene.rs` with a new `pub mesh: MeshId` field
  (imported from `mesh.rs`), re-exported from `mod.rs`/`lib.rs` at `Object`'s existing path so no
  caller-visible import path changes.

- [ ] **Step 1: Write the failing test for `resolve_texture_set`'s new rejection behavior**

Replace `resolve_none_and_out_of_range_fall_back_to_white`'s body (keep or rename the test — your
call, but its assertions must change) to assert: `None` still resolves to the white set (`Ok`); a
valid, correctly-owned `Some(TextureId)` resolves to the matching set (`Ok`); an out-of-range slot
with the correct owner returns `Err`; a mismatched owner with an otherwise in-range slot returns
`Err`. You will need a test-only way to construct a `TextureId` with a chosen owner/slot pair here —
add a `#[cfg(test)]` constructor or test helper in `texture.rs` for this (the production path only
ever creates one via `create_texture`).

- [ ] **Step 2: Run the test, confirm it fails**

Run: `cargo test --lib renderer::texture`
Expected: FAIL (or a compile error, since `resolve_texture_set`'s callers haven't been updated yet —
either outcome confirms you haven't implemented Step 3 yet).

- [ ] **Step 3: Implement the new `TextureId` shape and `resolve_texture_set`**

`resolve_texture_set`'s one production call site is inside `command.rs`'s `emit_object_draws`, a
plain `fn` (not `-> Result<()>`) called from both the real recording closure and CPU tests — it
cannot propagate a `?`. Per this plan's Global Constraints every task's own commit must still pass
the full gate suite, so in this same commit, patch that one call site to
`resolve_texture_set(...).expect("object handles are not yet validated before this point — this
placeholder is replaced by Task 4's up-front validation pass")` — a temporary, clearly-labeled
compile-preserving placeholder, exactly like spec 018's Task 1 used a literal `&[]` for the same
reason. This is safe for now because nothing calls `emit_object_draws` with attacker/user-supplied
invalid handles until a real caller exists — Task 4 replaces this `.expect` with real per-object
mesh/texture resolution once the up-front validation pass exists to guarantee it never panics in
practice. Do NOT restructure `emit_object_draws`/`ObjectCommand` itself in this task — that is
entirely Task 4's Step 5.

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cargo test --lib renderer::texture`
Expected: PASS.

- [ ] **Step 5: Create `scene.rs` and move `Object`**

Move `pub struct Object` (and nothing else) from `texture.rs` to a new `scene.rs`, adding `pub mesh:
MeshId`. Update `texture.rs`'s own doc comment for `Object` (currently describes it as "one drawn
copy of the uploaded mesh" — update the wording since there are now multiple meshes) — move that doc
comment to `scene.rs` alongside the relocated struct. Update `mod.rs`/`lib.rs`'s existing `Object`
export to point at the new path; no caller-visible import path should change (callers still write
`m2s2_rust_vulkan::Object` or whatever the crate root currently re-exports). `mesh.rs`'s module
declaration in `mod.rs` needs `pub(crate)` visibility (plain `mod mesh;` is unreachable from
`lib.rs`'s re-export) — this is a correction to Task 2's original wiring, make it here.

- [ ] **Step 5a: Patch EVERY existing `Object` literal in the crate with `MeshId::UNASSIGNED`**

`Object` now has a mandatory `mesh: MeshId` field with no `Default`, and this project's pre-commit
hook builds and tests the WHOLE crate on every commit — there is no way to land this task's commit
with any `Object { ... }` literal anywhere still missing that field, in ANY file, including this
crate's own library code. `mesh.rs` (Task 2, already landed) now exposes
`pub(crate) const MeshId::UNASSIGNED: MeshId` — a deliberately-invalid placeholder (`owner: 0`, never
allocated by `next_owner_id`) that compiles everywhere but can never resolve against any real
renderer. Add `mesh: MeshId::UNASSIGNED` to every one of the following, with NO other change to
each literal's existing fields:
- `src/renderer/mod.rs`'s `render_frame` method, which internally constructs one `Object` from its
  `model`/`camera` arguments (this file doesn't gain a `mesh` parameter until Task 4 — for THIS task,
  the placeholder is the correct, honest stand-in).
- `src/renderer/command.rs`'s own test fixture (the `Object` literal(s) in its `#[cfg(test)]`
  module).
- `src/renderer/headless.rs`'s test module — every existing `Object` literal (there are several:
  the determinism test, the self-test, the five point-light golden-image tests from spec 018).
- `examples/cube.rs`'s two existing `Object` literals.

This is a one-line addition per literal, nothing else changes about any of them. Task 4 replaces
every one of these placeholders: `mod.rs`'s own via `render_frame`'s new `mesh` parameter,
`command.rs`'s test fixture alongside that task's rewrite of the tests its restructuring breaks,
`cube.rs`'s via `create_mesh` as part of that same task's example migration, and (per Task 4's own
absorbed scope) `headless.rs`'s via that module's own new real mesh store — Task 4 is one atomic
commit covering all of this, since `command.rs`'s restructuring breaks both renderers' callers at
once (see Task 4's own note on why).

- [ ] **Step 6: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean, full stop — no deferred breakage anywhere. Same pass/ignored counts as the
branch baseline plus this task's new tests.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/texture.rs src/renderer/scene.rs src/renderer/command.rs src/renderer/mod.rs src/renderer/headless.rs src/lib.rs examples/cube.rs
git commit -m "feat(renderer): reject foreign/out-of-range TextureId; relocate Object to scene.rs with a mesh field"
```

---

### Task 4: Wire `VulkanRenderer`, restructure `command.rs`, migrate `triangle.rs`/`cube.rs`

**This task absorbs what was originally planned as two separate tasks.** Removing `vertices`/
`indices` from `with_window`/`unsafe new` breaks every caller's constructor CALL (wrong argument
count), not just a struct literal a placeholder can paper over — there is no `MeshId::UNASSIGNED`-
style stand-in for a missing function argument. Since this project's pre-commit hook builds and
tests the whole crate on every commit, the constructor signature change and both examples'
migration must land in the same commit. Keep the step numbering below as one continuous sequence;
do not split this into two commits.

**This task ALSO absorbs what was originally planned as part of a separate Task 5** (giving
`HeadlessRenderer` its own real mesh support), for the same reason: `command.rs`'s
`emit_object_draws`/`draw_frame` are shared machinery called by BOTH `VulkanRenderer` and
`HeadlessRenderer`. Changing their signature (Step 5 below) breaks `HeadlessRenderer::render_frame`'s
one call site too — and since `headless.rs`'s `#[cfg(test)]` module compiles into the SAME lib-test
binary as every other module's tests, that one broken call site blocks `cargo test --lib`/`--all-
targets` entirely, not just `headless.rs`'s own tests. There is no minimal placeholder for this (the
new signature needs a real, resolvable mesh store on both sides); giving `HeadlessRenderer` real mesh
support now is the only way to land this task's commit at all. The former Task 5 is now scoped down
to just writing new golden-image scenario tests against the mesh support this task adds, plus the
user handoff — see the (renumbered) Task 5 below.

**Files:**
- Modify: `src/renderer/mod.rs`
- Modify: `src/renderer/command.rs` (the real per-object mesh-bind restructuring)
- Modify: `examples/triangle.rs`, `examples/cube.rs` (migrate off the removed constructor
  parameters, replace Task 3's `MeshId::UNASSIGNED` placeholders with real `create_mesh`-derived
  handles)
- Modify: `src/renderer/headless.rs` (give `HeadlessRenderer` its own owner id + real mesh store,
  replacing its existing raw `Buffer::new_device_local` pair with `mesh::upload_mesh`; update every
  existing `Object` literal from the `MeshId::UNASSIGNED` placeholder to a real mesh id)

**Interfaces:**
- Consumes: Task 1's `next_owner_id`; Task 2's `validate_mesh_input`/`upload_mesh`/`resolve_mesh`/
  `StoredMesh`/`MeshId`; Task 3's strengthened `resolve_texture_set` and `Object` (now with `mesh:
  MeshId`) from `scene.rs`. Task 3 already patched `mod.rs`'s `render_frame` and `command.rs`'s test
  fixture with the temporary `MeshId::UNASSIGNED` placeholder so the crate kept compiling — this
  task replaces both placeholders with real values (Step 4 for `render_frame`, Step 6 for
  `command.rs`'s test), it is not discovering new breakage.
- Produces: `VulkanRenderer::with_window`/`unsafe new` with `vertices`/`indices` parameters removed;
  `pub fn create_mesh(&mut self, vertices: &[Vertex], indices: &[Index]) -> Result<MeshId>`;
  `pub fn render_frame(&mut self, mesh: MeshId, model: Mat4, camera: Camera) -> Result<()>` (gains
  the leading `mesh` parameter); `render_objects`/`render_objects_with_text` unchanged in signature,
  but their validation pass now also checks every object's `mesh`/`texture` handles.
  `command::FrameContext::draw_frame`/`record`'s signatures change: they stop taking a single shared
  `vertex_buffer`/`index_buffer`/`index_count`, and `ObjectCommand`/`emit_object_draws` gain a
  mesh-bind step resolved per object (consuming Task 2's `resolve_mesh` and Task 3's
  `resolve_texture_set`, both already proven-valid by the up-front pass — record-time resolution may
  `.expect("validated above")` rather than threading a second `Result` through the emission closure).

- [ ] **Step 1: Write the failing test for early mesh/texture validation rejecting the whole frame**

This needs a GPU-backed `VulkanRenderer` or `HeadlessRenderer` to exercise end-to-end, which isn't
available in this sandbox — write the test against the lowest GPU-free seam available (likely
`mesh.rs`'s/`texture.rs`'s own `resolve_*` functions, already covered in Tasks 2/3). If, while
implementing this task, you find a genuine non-GPU seam for testing `render_objects_with_text`'s
validation ORDER specifically (e.g. a pure function extracted from it that takes the object slice
and the stores and returns `Result<()>` before touching any renderer state), write it there and
note the seam in your report. If no such seam exists, say so plainly rather than fabricating a test
that doesn't exercise real behavior — this will be covered end-to-end by Task 5's GPU scenarios
regardless.

- [ ] **Step 2: Remove `vertices`/`indices` from both constructors and `init`; add `create_mesh`**

`VulkanRenderer` loses its `vertex_buffer: Option<Buffer>`/`index_buffer: Option<Buffer>` fields
entirely (and their construction/rollback/Drop-teardown code), replaced by `meshes:
Vec<mesh::StoredMesh>` (mirroring the existing `textures: Vec<StoredTexture>` field exactly — same
construction-time initialization to `Vec::new()`, same Drop-time `for mesh in
std::mem::take(&mut self.meshes) { mesh.destroy(...) }` pattern, same terminal-state
`std::mem::forget(std::mem::take(&mut self.meshes))` leak branch) and a new `owner: u64` field set
once via `next_owner_id()` in `init`. `create_mesh` calls `validate_mesh_input`, then `upload_mesh`,
then pushes the result and returns a `MeshId { owner: self.owner, slot: self.meshes.len() - 1 }` (or
however Task 2 shaped construction — `MeshId`'s fields are private to `mesh.rs`, so either expose a
`mesh.rs`-internal constructor this file calls, or push-then-build the id inside `mesh.rs` itself and
return it from `upload_mesh`/a wrapping function — your choice, whichever keeps `MeshId` genuinely
unconstructable from outside `mesh.rs`).

- [ ] **Step 3: Add the early mesh/texture validation pass**

In `render_objects_with_text`, extend the existing `for object in objects { validate_specular(...)?;
}` loop (or add an adjacent loop immediately before/after it, your call) to also call `mesh::
resolve_mesh(object.mesh, self.owner, &self.meshes)?` and, for `Some(texture)`, the texture
equivalent — both before the `pending_resize`/acquire check that follows. An invalid handle returns
`Err` immediately; nothing about `self` has been mutated yet at this point, satisfying "no partial
submission."

- [ ] **Step 4: Update `render_frame`'s signature**

Add the leading `mesh: MeshId` parameter; replace Task 3's `MeshId::UNASSIGNED` placeholder in its
internal synthesized `Object` with this real parameter.

- [ ] **Step 5: Restructure `command.rs`'s per-object draw emission**

Remove the single per-frame `cmd_bind_vertex_buffers`/`cmd_bind_index_buffer` call. Add a
`BindMesh(vk::Buffer, vk::Buffer)` (or however you choose to shape it) variant to `ObjectCommand`.
`emit_object_draws` loses its external `index_count: u32` parameter in favor of resolving each
object's own mesh (via `mesh::resolve_mesh`, `.expect`-able since the caller already validated) and
emitting a bind-mesh command before each object's existing bind-texture/push-model/draw-indexed
sequence, using THAT mesh's own index count for `DrawIndexed`. `draw_frame`/`record`'s parameter
lists lose the single `vertex_buffer: &Buffer, index_buffer: &Buffer` pair in favor of whatever
`emit_object_draws` now needs to resolve per object (likely `&[StoredMesh]` plus the owner id, or an
already-resolved slice — your call on the cleanest shape, consistent with how `texture_sets: &[vk::
DescriptorSet]` is already threaded through today).

- [ ] **Step 6: Rewrite the two command.rs tests this restructuring breaks**

`mixed_objects_bind_resolved_set_then_push_then_draw_in_order` and its `record_objects` helper
currently hardcode one shared `index_count` (36) for every object (and, since Task 3, carry a
temporary `MeshId::UNASSIGNED` placeholder in their `Object` fixtures) — per this plan's Global
Constraints, rewrite them to exercise interleaved different-mesh objects (mesh A, mesh B, mesh A
again) with real resolved meshes, replacing the placeholder entirely, and asserting each draw uses
its own mesh's bind calls and index count, in the original submission order. Also re-check
`empty_objects_emit_no_commands` still holds unmodified against the new per-object mesh-bind path
(per this plan's Review Focus) — it should, since an empty slice never enters the loop, but confirm
rather than assume.

- [ ] **Step 7: Update `submitted_indices`' computation**

Change from `objects.len() as u64 * index_buffer.len()` to a checked sum of each submitted object's
own resolved mesh's index count.

- [ ] **Step 8: Migrate `triangle.rs`**

Construct the renderer via `with_window(window, extent)` (no geometry args), then call
`create_mesh(&vertices, &indices)` once to get a `MeshId`, then pass that `MeshId` as `render_frame`'s
new leading argument at its existing call site. No visual change — same triangle, same camera.

- [ ] **Step 9: Migrate `cube.rs`**

Same pattern: construct without geometry, `create_mesh` once for the cube geometry already defined
in this file, and replace Task 3's `MeshId::UNASSIGNED` placeholder in both existing `Object`
literals with this real mesh id (both objects keep referencing the same single cube mesh — this is
still a migration, not a new demonstration; Task 5 is where a second, differently-shaped mesh gets
exercised, in a dedicated GPU test scene rather than this example). Keep every existing control
(`KeyCode::KeyH`/`KeyCode::F3`/camera movement/`KeyCode::KeyP` from spec 018) working unchanged.
This is `cube.rs`'s last reference to `MeshId::UNASSIGNED` anywhere in the crate — it is also the
LAST external (non-`mesh.rs`-test) reference of any kind, since `headless.rs`'s own references are
removed later in this same task (Step 11), not deferred to Task 5. Once this step is done, re-tighten
`UNASSIGNED`'s visibility in `mesh.rs` from `pub` back to `pub(crate)` (or tighter, if nothing else
needs it after Step 11 — check once both this step and Step 11 are done) and update its doc comment
accordingly, closing the temporary public-API crack Task 3 had to open only because
`examples/cube.rs` compiles as its own crate.

- [ ] **Step 10: Give `HeadlessRenderer` its own owner id and real mesh store**

`HeadlessRenderer::new`'s existing signature (`new(extent, vertices, indices)`) is `pub(crate)` test
infrastructure, not public API — it is NOT required to change by this spec, and this step should
NOT remove its `vertices`/`indices` parameters (that would be unrequested scope creep onto this
module's existing contract; callers still pass geometry at construction the same way). What DOES
need to change: give `HeadlessRenderer` its own `owner: u64` (via `next_owner_id()`, exactly like
`VulkanRenderer`'s own field from Step 2) and replace its existing two raw
`Buffer::new_device_local` calls (one for vertices, one for indices) with a single call to
`mesh::upload_mesh`, storing the result as a one-element `meshes: [mesh::StoredMesh; 1]` (or
`Vec<StoredMesh>` with exactly one entry pushed at construction — your choice, whichever reads more
naturally given this struct's existing field style). Expose the single mesh's id (e.g. a `fn
mesh(&self) -> MeshId` accessor, built once at construction via the same constructor `mesh.rs`
exposes to `mod.rs`) so call sites can reference it. Update `render_frame`'s one call into
`command.rs`'s restructured `emit_object_draws`/`draw_frame` to pass `self.owner` and `&self.meshes`,
matching whatever shape Step 5 settled on.

- [ ] **Step 11: Update every existing `Object` literal in `headless.rs`'s test module**

There are several (the determinism test, the self-test, and the five point-light golden-image tests
from spec 018) — replace each one's `mesh: MeshId::UNASSIGNED` placeholder with `mesh:
renderer.mesh()` (or your chosen accessor's name from Step 10). Confirm you haven't missed one by
running the build (Step 12) rather than grepping and trusting the count. This is `headless.rs`'s
last reference to `MeshId::UNASSIGNED` anywhere in the crate — once this step is done, the constant
has no remaining callers outside its own tests in `mesh.rs`; leave the constant itself in place
(Task 5 may still want it for a scenario test), but if `cargo clippy` flags it as unused outside
`#[cfg(test)]`, that confirms it's already correctly `#[cfg(test)]`-gated from Task 3's work and
needs no further change here.

- [ ] **Step 12: Run the full test suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean, full stop. No deferred breakage anywhere — this is the last task in the plan
that changes any shared renderer/command-recording code, so there is nothing left to defer.

- [ ] **Step 13: Commit**

```bash
git add src/renderer/mod.rs src/renderer/command.rs src/renderer/headless.rs examples/triangle.rs examples/cube.rs
git commit -m "feat(renderer): wire MeshId into VulkanRenderer and HeadlessRenderer, restructure per-object draw emission, migrate examples"
```

---

### Task 5: GPU golden-image acceptance tests, user handoff

**Files:**
- Modify: `src/renderer/headless.rs`

**Interfaces:**
- Consumes: Task 4's finished `HeadlessRenderer` (its own owner id, real mesh store, and every
  existing `Object` literal already pointing at a real mesh — nothing in this task's scope is still
  using `MeshId::UNASSIGNED`).
- Produces: nothing for later tasks — this plan's final task.

- [ ] **Step 1: Write the new golden-image multi-mesh scenario tests**

Following the existing `self_test_scene_matches_golden_with_zero_validation_errors` test's structure
(same fixed-camera helper, `assert_matches_golden` at tolerance 2, `renderer.validation_error_count()
== 0`), add `#[ignore]`d tests for the scenarios this spec's Test plan names that the existing
infrastructure doesn't yet cover:
- A frame with two objects referencing two DIFFERENT meshes (e.g. the existing cube plus a second,
  differently-sized/shaped mesh you construct inline) — proves per-object mesh binding actually binds
  the right buffers, not just that it compiles.
- Repeated use of the same mesh handle across multiple objects in one frame.
- A frame with zero meshes created but valid text-only content (confirms `HeadlessRenderer` doesn't
  require a mesh to exist to render — if `HeadlessRenderer`'s own constructor always takes geometry,
  note in your report whether this scenario is actually reachable given this file's constructor shape,
  or whether it's only meaningfully testable via `VulkanRenderer`, which this plan's executor also
  cannot run; a `#[ignore]`d placeholder noting the gap is acceptable if the scenario genuinely isn't
  constructible here).
Each needs its own distinct `#[ignore]` reason string naming its own golden PNG path under
`testdata/golden/` (per spec 018's ledgered Important finding — never reuse a byte-identical reason
string across two different tests, even if two tests legitimately share one golden file).

- [ ] **Step 2: Build and run the full gate suite**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo
build --all-targets && cargo test --all-targets`
Expected: all clean; every new/modified test compiles; GPU-dependent tests remain `#[ignore]`d, not
executed. The ignored-test count increases by however many new scenarios Step 1 added.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/headless.rs
git commit -m "test(renderer): multi-mesh golden-image acceptance tests for HeadlessRenderer"
```

- [ ] **Step 4: Hand off to the user — generate the new golden images**

This plan's executor has no GPU and cannot complete this step. Report clearly that the user must
run, on a machine with a Vulkan driver, one `M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless
-- --ignored <test_name>` invocation per new scenario from Step 1 to generate each golden PNG, then
run the same set of tests **without** `M2S2_UPDATE_GOLDEN` to confirm they pass, and finally re-run
the full pre-existing `#[ignore]`d suite (every test from specs 018/019) to confirm nothing
regressed. Only after the user confirms this should the golden PNGs be committed and spec 015's
Outcome section + `ROADMAP.md` be updated to reflect verified (not just implemented) status — the
same two-step handoff pattern specs 018/019 established.


## Completion follow-up

Task 5 is implemented and GPU-verified locally. The earlier no-GPU handoff assumption no longer
applies: three new baseline PNGs were generated and visually inspected, then all 15 ignored GPU tests
passed in comparison mode. Existing baselines were not regenerated. The headless fixture was extended
with real empty construction, additional mesh upload, text rendering, and early handle validation;
no placeholder test was used. A windowed probe verified counters, resize/minimize, upload between
frames, and shutdown. See spec 015 Outcome for final coverage and retained limitations.
