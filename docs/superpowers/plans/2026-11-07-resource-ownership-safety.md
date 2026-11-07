# Resource Ownership & Memory Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `VulkanRenderer`'s safe-Rust surface uphold its Vulkan lifetime and CPU memory-access contracts, so a consumer writing ordinary safe Rust cannot invalidate ownership, over-write mapped memory, double-free, or free GPU-in-use resources.

**Architecture:** Privatize the renderer and its owners behind re-exported public data types + two constructors (safe `with_window` retaining the window owner; `unsafe fn new` for raw handles). Replace `&self` unchecked mapped copies with exclusive bounded `&mut self` writes, repeatable `&mut self` destruction with ownership-consuming `destroy(self)`, `ManuallyDrop` teardown with `Option<T>` moves, and best-effort upload rollback with a submission-state machine that retains pending resources and enters a terminal leak state on unrecoverable wait failure.

**Tech Stack:** Rust, `ash` (Vulkan 1.3), `gpu-allocator` 0.28, `bytemuck`, `raw-window-handle`; `winit` (examples/dev only). No new runtime dependencies.

**Spec:** `specs/012-resource-ownership-and-memory-safety.md`

## Global Constraints

- **No new runtime dependencies.** No allocator replacement, no `Arc`/`Mutex` sprinkled through the renderer without demonstrated need.
- **Single-owner renderer** (resolved): the renderer solely owns device/instance/allocator/surface and all dependents; no independently-owned public GPU resources.
- Preserve rendering output, controls, and the existing per-frame-in-flight fence discipline (write a slot's mapped buffers only after waiting that slot's fence).
- gpu-allocator 0.28 `CpuToGpu` is `HOST_COHERENT`; **do not add flush calls** (not a defect on this path).
- Public data types stay available and re-exported from the crate root: `Vertex`, `Index`, `Object`, `TextureId`, `TextDraw`, `DirectionalLight`, `Diagnostics`, `RendererStats`, `RenderStatus`, `VulkanRenderer`, `init_logging`, and the `math` module. `stats()` is the only owner snapshot; add no mutable-handle getters.
- Quality gates must pass with **zero warnings** at every commit: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --all-targets`, `cargo test --all-targets`, and (new) `cargo test --doc` for compile-fail doctests.
- Conventional Commits; **no `Co-Authored-By`/AI-attribution trailer**.
- No shader or lighting-equation changes.

## Review Focus

- **Oversized mapped write** (`bytes.len() > logical capacity`) → `VulkanError::ValidationError` returned before any memory change; storage unchanged. *Test in Task 2 (`check_mapped_write` oversized case).*
- **Write to an unmapped buffer** → `ValidationError`, never a null-pointer deref. *Test in Task 2 (`check_mapped_write` unmapped case).*
- **Raw-handle construction without `unsafe`, or external code touching owner fields / constructing low-level owners** → fails to compile. *Compile-fail doctests in Task 1.*
- **Constructor failure partway through init** → only successfully-created resources are released, each exactly once, children before parents. *Fake-destruction injection test in Task 3.*
- **Fence wait failure after a successful submit** (upload or frame) → no pending staging/destination/command resource is freed before confirmed completion, and the terminal state blocks further reuse rather than waiting forever on an unsignaled fence. *Tests in Task 4.*

---

### Task 1: Encapsulation and the two constructors

Privatize the renderer and its low-level owners behind the re-exported public data types, and replace the single safe raw-handle `new` with a safe `with_window` (retaining the window owner) plus an `unsafe fn new`. Render output is unchanged; this is a pure surface/ownership-visibility change plus the window-owner field.

**Files:**
- Modify: `src/lib.rs` (make `renderer` non-public; keep/complete the `pub use` re-exports)
- Modify: `src/renderer/mod.rs` (private fields; `WindowOwner` trait + retained owner field; `with_window` + `unsafe fn new` sharing a private `init`; module visibilities)
- Modify: `src/renderer/{instance,device,swapchain,pipeline,command,allocator,buffer,depth,image,upload,per_frame_uniform,uniform,texture,text,light}.rs` (owner types/constructors `pub(crate)`, not `pub`; public data types stay `pub`)
- Modify: `examples/cube.rs`, `examples/triangle.rs` (construct via `with_window` with an `Arc` window)
- Test: compile-fail doctests in `src/renderer/mod.rs` (or `src/lib.rs`)

**Interfaces:**
- Consumes: existing `VulkanRenderer::new(display_handle, window_handle, extent, vertices, indices)` internals.
- Produces:
  - `pub(crate) trait WindowOwner: HasWindowHandle + HasDisplayHandle {}` with a blanket impl, plus `_window: Option<std::sync::Arc<dyn WindowOwner>>` field on `VulkanRenderer`. (No `Send`/`Sync` bound — the renderer never crosses threads and winit's `Window` is not guaranteed `Send`/`Sync` on every platform.)
  - `pub fn with_window<W: HasWindowHandle + HasDisplayHandle + 'static>(window: Arc<W>, window_extent: (u32,u32), vertices: &[Vertex], indices: &[Index]) -> Result<Self>`
  - `pub unsafe fn new(display_handle: RawDisplayHandle, window_handle: RawWindowHandle, window_extent: (u32,u32), vertices: &[Vertex], indices: &[Index]) -> Result<Self>`
  - A private `fn init(display_handle, window_handle, window_extent, vertices, indices, window: Option<Arc<dyn WindowOwner>>) -> Result<Self>` carrying today's construction body.

- [ ] **Step 1: Write compile-fail doctests for the encapsulation boundary**

In `src/renderer/mod.rs`, add `compile_fail` doctests on `VulkanRenderer`: one that calls `m2s2_rust_vulkan::renderer::instance::VulkanInstance::new()` (path/type no longer reachable), one that reads a renderer field (e.g. `r.device`), and one that calls `VulkanRenderer::new(..)` **without** an `unsafe` block. Each is a ```` ```compile_fail ```` doc block with a one-line comment naming what it proves.

- [ ] **Step 2: Run to verify they currently compile (so the test is real)**

Run: `cargo test --doc 2>&1 | tail -20`
Expected: the doctests **fail the suite** because the code inside them *compiles today* (fields are `pub`, `new` is safe) — `compile_fail` blocks that compile are reported as failures. This confirms the tests exercise the real restriction before it exists.

- [ ] **Step 3: Privatize owners and re-export only data types**

In `src/lib.rs`, change `pub mod renderer;` to `mod renderer;` and ensure every public item is covered by a `pub use` (the list in Global Constraints). In `src/renderer/mod.rs` make every `VulkanRenderer` field non-`pub` (`pub(crate)` or private); keep the owner submodules usable internally but make their owner **types** and `new`/low-level constructors `pub(crate)` (not `pub`). Leave public data types (`Vertex`, `Index`, `Object`, `TextureId`, `TextDraw`, `DirectionalLight`, diagnostics types) `pub` so the `pub use` re-exports resolve.

- [ ] **Step 4: Add the window owner + both constructors**

Add the `WindowOwner` trait + blanket impl and the `_window` field. Move the current `new` body into private `init(.., window: Option<Arc<dyn WindowOwner>>)`, storing `window` in `_window`. Implement `with_window` (extract handles from the `Arc`, clone it into `init` as `Some`) and `unsafe fn new` (pass the raw handles and `None`) with the SAFETY doc from the spec. Surface extraction from the retained owner happens once at construction.

- [ ] **Step 5: Update examples to the safe path**

In `examples/cube.rs` and `examples/triangle.rs`, wrap the winit `window` in `Arc`, pass a clone to `VulkanRenderer::with_window(...)`, and keep a clone for the event loop. Remove the old `window.display_handle()/window_handle()` raw-handle plumbing from the construction call.

- [ ] **Step 6: Run the compile-fail doctests and gates**

Run: `cargo test --doc` then `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: doctests pass (restrictions now hold); all gates clean; existing 54+2 tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs src/renderer/*.rs examples/cube.rs examples/triangle.rs
git commit -m "refactor(renderer)!: privatize owners; add safe with_window and unsafe new"
```

---

### Task 2: Bounded, exclusive mapped writes

Replace `Buffer::write_mapped(&self, &[u8])` (unchecked, shared) with an exclusive, length-checked `&mut self` write returning `Result<()>`, and thread `&mut` through the camera/light/text update chain. The capacity check is a free function so it is unit-testable without a GPU.

**Files:**
- Modify: `src/renderer/buffer.rs` (`logical_capacity` field; `check_mapped_write`; `write_mapped(&mut self, &[u8]) -> Result<()>`; set capacity in both constructors)
- Modify: `src/renderer/per_frame_uniform.rs` (`update(&mut self, ..) -> Result<()>`)
- Modify: `src/renderer/uniform.rs`, `src/renderer/light.rs` (wrappers `update(&mut self, ..) -> Result<()>`)
- Modify: `src/renderer/text.rs` (text uniform/vertex update `&mut self -> Result<()>`)
- Modify: `src/renderer/command.rs` (`draw_frame` takes `&mut Uniforms`, `&mut Light`, `&mut TextRenderer`; propagate `?`)
- Modify: `src/renderer/mod.rs` (pass `&mut self.uniforms`, `&mut self.light`, `&mut` text renderer into `draw_frame`)

**Interfaces:**
- Consumes: `Allocation::mapped_slice_mut()`.
- Produces:
  - `fn check_mapped_write(logical_capacity: usize, mapped: bool, len: usize) -> Result<()>` (private, pure).
  - `Buffer::write_mapped(&mut self, bytes: &[u8]) -> Result<()>`.
  - `PerFrameUniform::update(&mut self, frame: usize, bytes: &[u8]) -> Result<()>`; `Uniforms::update(&mut self, frame, &CameraUniforms) -> Result<()>`; `Light::update(&mut self, frame, &DirectionalLight) -> Result<()>`.

- [ ] **Step 1: Write failing tests for `check_mapped_write`**

In `src/renderer/buffer.rs` tests:

```rust
#[test]
fn mapped_write_bounds_and_mapping() {
    assert!(check_mapped_write(16, true, 0).is_ok());   // empty
    assert!(check_mapped_write(16, true, 16).is_ok());  // exact
    assert!(check_mapped_write(16, true, 8).is_ok());   // smaller
    assert!(check_mapped_write(16, true, 17).is_err()); // oversized
    assert!(check_mapped_write(16, false, 4).is_err()); // unmapped
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib renderer::buffer`
Expected: FAIL — `check_mapped_write` does not exist.

- [ ] **Step 3: Implement capacity + checked exclusive write**

Add `logical_capacity: usize` to `Buffer`, set it in `new_host_visible_mapped` (to `size as usize`) and `new_device_local` (to the uploaded byte length). Implement `check_mapped_write` (oversized or unmapped → `VulkanError::ValidationError`, else `Ok`). Rewrite `write_mapped(&mut self, bytes)` to call `check_mapped_write(self.logical_capacity, self.allocation.as_ref().and_then(|a| a.mapped_ptr()).is_some(), bytes.len())?`, then copy via `self.allocation.as_mut().unwrap().mapped_slice_mut().unwrap()[..bytes.len()].copy_from_slice(bytes)`. Update the doc comment (drop the "`&self` is fine" note).

- [ ] **Step 4: Thread `&mut` + `?` through the update chain**

Make `PerFrameUniform::update`, `Uniforms::update`, `Light::update`, and the text updates take `&mut self` and return `Result<()>`. In `command.rs` `draw_frame`, change the `uniforms`/`light`/text parameters to `&mut`, and `?`-propagate the writes (performed after the fence wait, as today). In `mod.rs`, pass `&mut self.uniforms`, `&mut self.light`, and the `&mut` text renderer.

- [ ] **Step 5: Run buffer tests + full gates**

Run: `cargo test --lib renderer::buffer && cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/renderer/buffer.rs src/renderer/per_frame_uniform.rs src/renderer/uniform.rs src/renderer/light.rs src/renderer/text.rs src/renderer/command.rs src/renderer/mod.rs
git commit -m "refactor(renderer): exclusive bounded mapped writes"
```

---

### Task 3: Ownership-consuming destruction and `Option` teardown

Convert allocator-aware `destroy(&mut self, ..)` to `destroy(self, ..)` so cleanup can't run twice, replace the `ManuallyDrop` members with `Option<T>` moved out during teardown, and route every constructor/resize error path through release of only the resources that were created. Build a tiny fake-destruction recorder to assert order/counts and single-release without a GPU.

**Files:**
- Modify: `src/renderer/buffer.rs`, `depth.rs`, `image.rs`, `texture.rs`, `per_frame_uniform.rs`, `uniform.rs`, `light.rs` (`destroy(self, ..)`)
- Modify: `src/renderer/mod.rs` (`Option<T>` owner fields; teardown via `take()` + moves; constructor/resize rollback audit)
- Create: `src/renderer/teardown.rs` (a `Teardown` ordering helper + a test-only recorder), or put the recorder in a `#[cfg(test)]` module of `mod.rs`
- Test: ordering/single-release tests; a `compile_fail` doctest for double-destroy

**Interfaces:**
- Produces: `destroy(self, device: &VulkanDevice, allocator: &mut Allocator)` on each allocator-backed owner; a pure, testable teardown-order routine over a sequence of release closures that records calls.

- [ ] **Step 1: Write the double-destroy compile-fail doctest + an ordering/single-release unit test**

Compile-fail doctest: construct a stub owner, call its consuming `destroy(..)` twice — second call must not compile (value moved). Ordering test: drive the teardown routine with a recorder of fake resources and assert (a) each resource released exactly once, (b) children before parents, (c) an injected failure at resource *k* releases exactly resources `0..k`, none twice.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib renderer:: && cargo test --doc`
Expected: FAIL/non-compile — consuming `destroy` and the teardown routine don't exist yet.

- [ ] **Step 3: Make `destroy` consuming across allocator-backed owners**

Change each `destroy(&mut self, ..)` to `destroy(self, ..)`, taking fields by value (no `self.x.take()` dance needed once `self` is owned). A buffer/image destroys its handle then returns its allocation to the allocator exactly once.

- [ ] **Step 4: Replace `ManuallyDrop` with `Option<T>`; audit rollback paths**

In `mod.rs`, change `instance`/`device`/`swapchain`/`pipeline`/`frame_context`/`allocator` (and the buffers/uniforms/light/depth/textures/white-texture) to `Option<T>` (or keep direct owned where drop-order allows), and rewrite `Drop` to `take()` and move each owner into its consuming `destroy`/drop in the spec's teardown order (frame/sync → pipelines/layouts → descriptor pools/sets/views/samplers/buffers/images → allocations → allocator → swapchain views/swapchain/surface → device → instance → retained window owner last). Audit `init`'s partial-init rollback and the resize replacement path so each releases only successfully-created resources, replacements are built before old resources are consumed, and no destructor panics.

- [ ] **Step 5: Run gates**

Run: `cargo test --doc && cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/renderer/*.rs
git commit -m "refactor(renderer): ownership-consuming destruction and Option teardown"
```

---

### Task 4: Submission-aware upload ownership and terminal failure state

Give the renderer a named submission state and make `submit_once` (and the frame path's fence waits) retain all pending resources until completion is confirmed, entering a terminal "poisoned" leak state on an unrecoverable wait failure instead of freeing possibly-in-use memory or hanging on a reset fence.

**Files:**
- Modify: `src/renderer/upload.rs` (retain pool/fence/staging/destination until a successful wait; on wait failure, surface a terminal error without freeing)
- Modify: `src/renderer/mod.rs` (a `RenderState { Active, Terminal }` (or richer `Submission` enum) field; render/upload/resize entry points return terminal error when poisoned; teardown leaks the whole owner graph when terminal)
- Modify: `src/renderer/command.rs` (frame path: if recording/submission fails after fence reset, transition to terminal rather than permitting a retry that waits on an unsignaled fence)
- Modify: `src/renderer/error.rs` (a `VulkanError` variant for terminal/device-sync failure, if none fits)
- Test: submission-state transition tests over fake submit/wait outcomes

**Interfaces:**
- Produces: an internal submission/render-state type with states `not-submitted | submitted-pending | completed | terminal`, a pure transition function tested against injected submit/wait results, and a renderer guard that short-circuits public calls when terminal.

- [ ] **Step 1: Write failing tests for the state machine**

Assert: before-submit failure → ordinary rollback allowed (not terminal); submit success + wait success → completed (resources releasable); submit success + wait failure → terminal, and the transition reports "do not free pending / do not retry"; once terminal, every subsequent transition stays terminal.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib renderer::`
Expected: FAIL — the state type/transition don't exist.

- [ ] **Step 3: Implement the submission state + upload retention**

Add the state type + transition. In `upload.rs`, keep the `Cleanup` guards for the pre-submit window, but after a successful `queue_submit` hold pool/fence/staging/destination under one pending owner; free them only after a successful `wait_for_fences`. On wait failure, return the terminal error and do not free. Add the `VulkanError` variant if needed.

- [ ] **Step 4: Wire the terminal state into the renderer**

Add the state field to `VulkanRenderer`; `render_*`/`create_texture`/`resize` return the terminal error immediately when poisoned. In the frame path, a post-fence-reset recording/submission failure or a fence wait failure transitions to terminal (no retry on an unsignaled fence). In `Drop`, when terminal, **skip destruction and leak the whole owner graph** (including allocator, device, instance, and the retained `_window` owner); otherwise run the Task 3 ordered teardown. Log the terminal transition once.

- [ ] **Step 5: Run gates**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets && cargo test --doc`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/renderer/upload.rs src/renderer/mod.rs src/renderer/command.rs src/renderer/error.rs
git commit -m "feat(renderer): submission-aware upload ownership and terminal failure state"
```

---

### Task 5: Docs, safety comments, spec outcome, GPU acceptance

Finish the boundary documentation and record results.

**Files:**
- Modify: `README.md` (if it shows construction), `examples/*` headers, safety comments at `unsafe fn new` and the terminal-leak path
- Modify: `CLAUDE.md` Architecture note (safe `with_window` softens no-window-ownership only on the opt-in path)
- Modify: `specs/012-resource-ownership-and-memory-safety.md` (Outcome), `ROADMAP.md` (note the safety cleanup landed)

- [ ] **Step 1: Update prose + safety comments**

Document both constructors (safe default vs. `unsafe` escape hatch), the bounded-write/consuming-destroy contracts, and the terminal-leak policy where they live. Update the `CLAUDE.md` Architecture paragraph.

- [ ] **Step 2: Verify gates**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets && cargo test --doc`
Expected: all clean.

- [ ] **Step 3: GPU acceptance (outside sandbox, user machine)**

Run: `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo run --example cube` (and `triangle`).
Expected: unchanged output/controls; dynamic light updates, text hidden/visible/empty, repeated resize/minimize/restore, texture creation between frames, and normal shutdown produce **no validation/synchronization errors and no allocator leak reports** on success paths.

- [ ] **Step 4: Record outcome + commit**

Fill the spec Outcome (what landed, test counts, gates, GPU result) and the ROADMAP note.

```bash
git add specs/012-resource-ownership-and-memory-safety.md ROADMAP.md CLAUDE.md README.md examples/*.rs
git commit -m "docs: record spec 012 outcome and safety-boundary documentation"
```
