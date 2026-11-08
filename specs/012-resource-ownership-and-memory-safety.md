# 012: Resource ownership and memory safety cleanup

**Status**: Implemented and verified (CPU gates + GPU acceptance clean). See Outcome.
**Roadmap item**: [Engine-level concerns](../ROADMAP.md#phase-5-engine-level-concerns), pulled forward
before adding more rendering features.
**Prerequisite**: Current renderer through spec 011, including its camera/light validation amendment.

## Goal

Make the renderer's safe Rust interface uphold its Vulkan lifetime and CPU memory-access contracts.
Prevent callers from separating resources from their owners; replace unchecked mapped copies and
repeatable destruction with bounded writes and ownership-consuming cleanup. Preserve rendering output.

Success means fewer safety obligations depend on comments or remembering a destroy order, with the
remaining unsafe obligations explicit and testable. Removing `unsafe` keywords alone is not success.

## Findings motivating this work

| Current location | Pitfall | Proposed boundary |
| --- | --- | --- |
| `VulkanRenderer` public owner fields | Safe replacement/swapping can invalidate parent/child relationships | Private owners; public operations and value snapshots |
| Public low-level constructors | Returned children do not retain borrowed Vulkan parents | Internal constructors owned by one renderer |
| Safe constructors taking raw window handles | Window/display validity and lifetime cannot be checked | Explicit unsafe raw-handle constructor contract |
| `Buffer::write_mapped(&self, bytes)` | Unchecked length and writes through shared access | Exclusive, bounded writes using mapped slices |
| `destroy(&mut self, ...)` | Allocation is taken, but handles remain reusable | Consuming cleanup and `Option::take()` at ownership sites |
| Upload wait failure after successful submission | Rollback can free commands and memory still in use | Submission-aware ownership of all pending resources |

The current frame path already waits on the slot fence before mapped updates. Preserve that discipline.
The installed gpu-allocator 0.28 Vulkan backend requires HOST_COHERENT for CpuToGpu allocations, including
fallbacks. Missing flush calls are not a current defect on that path; this spec does not add flushes
without a change to the allocation contract.

## Non-goals

New rendering behavior, an allocator replacement, texture unloading, bindless resources, a render graph,
parallel submission, a general resource registry, or device-loss recovery/reinitialization. Do not add
Arc/Mutex ownership throughout the renderer without a demonstrated need. No new runtime dependencies.

The library continues to leave window/event-loop creation to its caller. Vulkan driver operations
remain unsafe internally; this is not a claim that the driver or arbitrary GPU execution is verified
by Rust's borrow checker.

## Public API and compatibility

This is an intentional pre-1.0 API tightening. Existing data types and high-level rendering methods stay
available: Vertex, Index, Object, TextureId, TextDraw, DirectionalLight, diagnostics, render methods,
resize, texture creation, stats, and the fallible light setter introduced in the spec 011 amendment.

### Encapsulation

Every VulkanRenderer field becomes private. Make low-level Vulkan owner types/constructors internal,
including instance, device, allocator, swapchain, pipeline, frame context, and GPU buffer ownership.
Public modules may remain as data-type namespaces where needed, but must not expose those owners.
`stats()` remains the supported value snapshot; do not add replacement mutable-handle getters.

### Window construction: decision (resolved 2026-11-07)

Provide **both** constructors, sharing one internal constructor. The safe one is the default path for
consumers used to safe-Rust semantics; the unsafe one is the escape hatch for consumers that own their
own window/event loop and do not want the library to hold a window reference.

1. **Safe `with_window`** — upholds the window-lifetime contract *by construction* by retaining the
   window/display owner for the renderer's lifetime, so no caller `unsafe` is needed:

```rust
impl VulkanRenderer {
    /// Safe: the renderer retains shared ownership of `window`, so the surface's window/display
    /// connection stays alive until after the renderer has destroyed its surface — including the
    /// terminal-leak path below. `window` provides both handles. winit callers keep their own clone.
    pub fn with_window<W: HasWindowHandle + HasDisplayHandle + 'static>(
        window: std::sync::Arc<W>,
        window_extent: (u32, u32),
        vertices: &[Vertex],
        indices: &[Index],
    ) -> Result<Self>;
}
```

2. **`unsafe fn new`** — raw handles, no retained owner, for the no-ownership consumers:

```rust
impl VulkanRenderer {
    /// # Safety
    /// Handles must describe a valid, matching window/display connection. Their owners must stay
    /// alive until the renderer has completed GPU use and destroyed its surface. The caller must
    /// satisfy platform thread-affinity requirements. See failure/leak policy below.
    pub unsafe fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_extent: (u32, u32),
        vertices: &[Vertex],
        indices: &[Index],
    ) -> Result<Self>;
}
```

The renderer retains the safe path's window via a type-erased owner (e.g. `Arc<dyn WindowOwner>`, a
small internal trait combining the two handle traits — exact mechanism is the plan's call; do **not**
make `VulkanRenderer` generic over `W`). The retained owner makes the lifetime relationship real, so
the safe API does not merely move an unverifiable contract into documentation. Dropping/leaking the
renderer must also drop/retain this owner (see teardown and the terminal-leak policy).

This softens the "library owns no window" architecture principle **only on the opt-in safe path**; the
`unsafe fn new` path preserves strict no-ownership. Update examples: `cube.rs`/`triangle.rs` adopt the
safe `with_window` path (holding the window in an `Arc`); keep one example or doc snippet exercising
`unsafe fn new` with a concrete SAFETY comment. Audit capture/destruction order rather than assuming
closure capture order preserves the window.

## Internal ownership and Vulkan objects

### One renderer owns the resource graph

Keep one internal owner for device, instance/loader, allocator, surface, and all their dependent
resources. These objects must not be independently constructible or replaceable by external safe code.
An ash::Device or ash::Instance clone copies access to handles; it does not keep the Vulkan object alive.

Replace ManuallyDrop members used for ordered teardown with private Option<T> members where teardown
needs to move values out. Use `take()` and ordinary Rust moves/drops instead of ManuallyDrop::drop.
Only the central owner performs allocator-aware destruction. Do not claim that Option alone makes the
GPU lifetimes safe; its purpose is to represent ownership transfer without exposing destroyed values.

For allocator-backed objects, consume ownership:

```rust
// Representative internal signatures, not additional public API.
impl Buffer {
    fn write_mapped(&mut self, bytes: &[u8]) -> Result<()>;
    fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator);
}
// Apply the same consuming pattern to depth/images, uniforms, and stored textures.
```

Constructor rollback owns partial resources in Options or scoped owners and releases only resources
whose creation succeeded. A successful constructor transfers ownership exactly once. Resize creates
replacements before consuming old resources and runs only after device completion is established.
Do not fix repeatable destruction by merely nulling handles while leaving ambiguous ownership.

### Teardown order

After completion has been established:
1. Release frame command buffers/pools and synchronization objects.
2. Release scene/text pipelines and their pipeline layouts.
3. Release descriptor pools, sets, and layouts; release dependent views/samplers, buffers, and images.
4. Return every buffer/image allocation to the allocator, after destroying the bound object.
5. Drop allocator backing memory.
6. Release swapchain views, swapchain, then surface.
7. Destroy device, then instance/debug objects in their valid order, retaining the loader until last.

Exact sibling ordering may differ where Vulkan allows it. The invariants are that pending work cannot
reference released objects, bound allocations outlive their images/buffers, and parents outlive children.
No destructor should panic. Allocation-free errors must be surfaced/logged deliberately, not silently
presented as successful cleanup. Do not add an automatic second free after ownership has been consumed.

## Mapped memory access

Store the requested logical byte capacity in Buffer, separate from element count and allocator padding.
Use Allocation::mapped_slice_mut() plus a checked slice and copy_from_slice(), with `&mut self` through
the camera/light/text update chain. Reject writes above logical capacity before changing memory. Return
a validation error for attempts to map an unmapped buffer; do not dereference an absent pointer.

A mutable CPU borrow prevents overlapping host writes through this API. It does not establish GPU
completion. Keep frame-slot selection and mapped updates inside the renderer: wait successfully for
that slot's fence, write only that slot's buffers, record, and submit. Upload staging buffers follow
the separate upload ownership rules below. Do not expose mapped references beyond their intended scope.

If writes become fallible, perform them before acquisition/fence reset where possible. Any error after
acquisition must also account for the acquired image and signaled semaphore; avoiding an unsignaled
fence alone is insufficient. Define a terminal frame-error state if retry cannot be made correct.

## Submission and failure handling

Use a named internal submission state distinguishing not-submitted, submitted/pending, completed, and
terminal failure. This is a small ownership boundary, not a general command scheduler.

Before successful queue submission, ordinary rollback may immediately release the upload resources.
After successful submission, one pending owner retains the command pool, fence, staging buffer, and
destination resource/allocation. Retaining only the fence/pool is insufficient: an error must not let
the caller destroy the staging or destination object while the copy can still execute.

A wait failure is not a completion certificate. Preserve the pending owner until a successful wait or
idle operation establishes completion, or follow the specifically documented device-loss teardown rules.
The renderer must not continue ordinary frame/upload work after a terminal synchronization error.
Likewise, if recording/submission fails after a frame fence is reset, do not permit a retry that waits
forever on a fence that no submission will signal.

**Terminal failure policy (resolved 2026-11-07):** if completion cannot be established and the device
is not confirmed lost, the renderer enters a **terminal "poisoned" state and deliberately leaks the
entire live ownership graph** — allocator, device, instance/loader, and (for the safe constructor) the
retained window/display owner. It does **not** run destruction (freeing GPU-in-use memory is worse
than leaking), and it does **not** abort the host process (a library must not kill its caller). It logs
the error once, and every subsequent `render_*`/`create_texture`/resize call returns a terminal error
rather than retrying or waiting forever on a fence no submission will signal. Leaking only child
handles is not acceptable — the leak must retain the whole graph so nothing the GPU may still touch is
freed. For the `unsafe fn new` path the renderer does not own the window, so its SAFETY contract
already requires the caller keep the window alive; the terminal state does not change that obligation.

## Interactions and implementation order

1. Restrict owner visibility and document/update raw construction at the public boundary.
2. Add logical capacities and exclusive checked mapped writes; update camera/light/text/staging callers.
3. Convert allocator-aware destruction and central teardown to ownership-consuming operations; audit
   every constructor and resize error path. Preserve the existing shared allocator/upload helpers.
4. Introduce pending-submission ownership and the agreed failure policy. Audit frame-error retry rules.
5. Update README/examples, safety comments, and verification notes. Run all gates after each step.

Primary modules: mod, instance, device, allocator, buffer, depth, image, texture, per_frame_uniform,
uniform, light, text, command, upload, and pipeline. No shaders or lighting equations need to change.

## Test plan and acceptance criteria

CPU-only checks:
- Compile-fail documentation tests show external code cannot access/replace owner fields or construct
  low-level owners. Raw-handle construction without unsafe fails to compile.
- Bounded-write tests cover empty, exact-capacity, smaller, oversized, and unmapped writes. Oversized
  writes leave storage unchanged; allocator padding is not counted as logical capacity.
- Use the actual cleanup/ownership implementation with fake destruction callbacks or a minimal test
  backend to record order/counts. Inject constructor failures at resource-creation boundaries; verify
  completed children are released exactly once before parents. Do not test a parallel simulated design.
- Consuming cleanup cannot be called twice on the same owner; resize failure retains valid old ownership.
- Inject failure before submission, during submission, and after submission while waiting. Verify no
  pending resource is destroyed before confirmed completion, and terminal failure prevents reuse.
- Cover frame failures around acquire, recording, fence reset, and submission; retry must not hang or
  reuse a signaled binary semaphore incorrectly.

Run cargo fmt --all -- --check, cargo clippy --all-targets --all-features -- -D warnings,
cargo build --all-targets, cargo test --all-targets, and cargo test --doc. Compile-fail tests must fail
for the intended restriction, not an unrelated missing import/type error. Use Miri for compatible pure
CPU ownership tests if available; it does not validate Vulkan driver execution or GPU synchronization.

GPU acceptance, opt-in outside normal CI: triangle and lit/textured cubes; dynamic light updates;
text hidden/visible and empty scenes; repeated resize/minimize/restore; texture creation between frames;
normal shutdown. Require no validation/synchronization errors and no allocator leak reports on successful
paths. Error-path tests must establish the chosen retention policy rather than expecting ordinary
cleanup on unconfirmed completion. Verify output/controls remain unchanged.

## Resolved decisions (2026-11-07)

- **Constructor:** provide both a safe `with_window` (retains the window owner — the default) and an
  `unsafe fn new` (raw handles, no ownership). See "Window construction".
- **Terminal wait-error policy:** poisoned terminal state that leaks the whole ownership graph; no
  destruction of possibly-in-use memory, no host abort, terminal error on further calls. See
  "Submission and failure handling".
- **Single-owner renderer:** yes — the renderer is the sole supported owner of the GPU resource graph;
  independently-owned public GPU resources are out of scope (would need a separate
  parent-ownership/threading design).

## Open questions (implementation detail, not design-blocking)

- Which minimal test backend exercises the actual cleanup/failure paths without a real GPU (fake
  destruction callbacks vs. a trait-abstracted device recorder)? Decide during planning/implementation.

## Outcome

Implemented across four tasks on `feat/012-resource-ownership` (plan
`docs/superpowers/plans/2026-11-07-resource-ownership-safety.md`):

1. **Encapsulation + constructors.** `mod renderer` is now private; only the public data types and
   `VulkanRenderer` are re-exported from the crate root (plus `RawDisplayHandle`/`RawWindowHandle` so
   the unsafe constructor's signature is nameable). All `VulkanRenderer` fields are private. Added the
   safe `with_window<W: HasWindowHandle + HasDisplayHandle + 'static>(Arc<W>, …)` (retains the window
   via `Arc<dyn WindowOwner>`) and `unsafe fn new(raw handles, …)`, sharing a private `init`. Three
   `compile_fail` doctests prove external code cannot construct the low-level owners, read a private
   field, or call `new` without `unsafe`. Examples use `with_window`.
2. **Bounded exclusive mapped writes.** `Buffer::write_mapped(&mut self, &[u8]) -> Result<()>` with a
   pure, unit-tested `check_mapped_write` (empty/exact/smaller/oversized/unmapped); a `logical_capacity`
   field distinct from element count/padding; `&mut` + `?` threaded through the camera/light/text
   update chain.
3. **Ownership-consuming destruction + `Option` teardown.** Every allocator-backed owner's
   `destroy(self, ..)` consumes `self` (a second destroy cannot compile); the six allocator-backed
   renderer fields became `Option<T>` moved out via `take()` in `Drop`/resize, and `textures` via
   `mem::take`.
4. **Submission-aware ownership + terminal state.** A pure `SubmitState`
   (NotSubmitted→Pending→Completed/Terminal) decides when pending resources may be released; `upload`
   leaks its transient pool/fence on a wait failure; the frame path maps the step-1 wait and all
   post-reset failures to `VulkanError::Terminal`; the renderer then rejects all public calls and, in
   `Drop`, deliberately leaks the whole resource graph (no free, no abort).

Resolved decisions are recorded above. **Scoped deviations (ledgered):** the six core owners
(`instance`/`device`/`swapchain`/`pipeline`/`frame_context`/`allocator`) remain `ManuallyDrop` rather
than `Option` — converting them is pure `unsafe`-reduction (they drop via their own `Drop`, with no
double-free benefit) at ~46 access sites and is not reachable by safe consumers, so it is **deferred**.
`upload::submit_once` leaks its own pool/fence on a wait failure, but the caller's staging/destination
rollback on that terminal error is unchanged; the realistic trigger is device loss (infinite wait), so
this is a low-risk residual, also **deferred**.

Verified (CPU): 56 lib + 2 example unit tests and 3 `compile_fail` doctests pass; `cargo fmt`,
`clippy -D warnings`, `build --all-targets`, `test --all-targets`, and `test --doc` are clean.

GPU acceptance: both examples (`cube` and `triangle`, now constructed via the safe `with_window`
path) were run with synchronization validation enabled and produced **no validation or synchronization
errors** — confirming the ownership/teardown rework preserved correct output and sync behavior.

## References

- [ash-window surface safety contract](https://docs.rs/ash-window/0.13.0/ash_window/fn.create_surface.html)
- [Rust field destruction order](https://doc.rust-lang.org/reference/destructors.html)
- [Vulkan fence wait results](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForFences.html)
- [Command-pool destruction requirements](https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyCommandPool.html)
