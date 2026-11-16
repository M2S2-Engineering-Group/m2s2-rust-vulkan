# 015: Multiple mesh resources

**Status**: Implemented and verified (CPU gates, headless GPU suite, and windowed smoke test pass).
**Roadmap items**: Phase 3 resource handles; Phase 2 scenes containing different geometry.
**Dependencies**: Spec 012's private ownership/cleanup contract and spec 013's Camera and
SpecularMaterial contracts — both implemented and merged; no longer provisional.
Spec 014 mipmaps are independent.

## Goal

Upload several independent indexed meshes and reference them through typed renderer-owned handles.
Render different geometry in one frame without constructing another renderer or pipeline.

## Non-goals

Model loading, scene graphs, mesh editing, unloading/slot reuse, shared buffers/suballocation within
mesh buffers, 32-bit indices, culling, materials as GPU resources, or automatic draw reordering.

## Public API: proposed pre-1.0 migration

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshId { /* private renderer-owner identity and slot */ }

pub struct Object {
    pub mesh: MeshId,
    pub model: Mat4,
    pub texture: Option<TextureId>,
    pub specular: SpecularMaterial,
}

impl VulkanRenderer {
    // Both existing constructors (spec 012) lose their vertices/indices parameters.
    pub fn with_window<W: HasWindowHandle + HasDisplayHandle + 'static>(
        window: Arc<W>, window_extent: (u32, u32),
    ) -> Result<Self>;
    pub unsafe fn new(
        display_handle: RawDisplayHandle, window_handle: RawWindowHandle, window_extent: (u32, u32),
    ) -> Result<Self>;
    pub fn create_mesh(&mut self, vertices: &[Vertex], indices: &[Index]) -> Result<MeshId>;
    pub fn render_frame(&mut self, mesh: MeshId, model: Mat4, camera: Camera) -> Result<()>;
    pub fn render_objects(&mut self, objects: &[Object], camera: Camera) -> Result<()>;
    pub fn render_objects_with_text(
        &mut self, objects: &[Object], camera: Camera, text: &[TextDraw<'_>],
    ) -> Result<()>;
}
```

Remove geometry arguments from renderer construction. Examples construct a renderer, upload meshes,
and put the returned MeshId in each Object. No implicit default mesh or hidden upload on first draw.
A renderer with no meshes can still clear/present and draw text. Vertex and Index remain unchanged;
Index is u16, and both index and vertex byte counts must fit the chosen upload/count representations.

`render_frame` uses the selected mesh, white-default texture, and default specular material. Texture
creation and the directional-light API otherwise remain unchanged.

## Handle identity and validation

Use one private renderer identity shared by MeshId and TextureId. A shared internal owner/slot handle
representation prevents two implementations of the same identity check; distinct public newtypes keep
mesh/texture arguments non-interchangeable. Only the producing renderer can resolve its handles.

This intentionally strengthens TextureId too: reject foreign or out-of-range Some(handle) rather than
silently binding white. None is the only explicit request for the white default. A foreign handle with
the same numeric slot must never resolve to a local resource. Do not expose constructors or raw slots.
Renderer identities must not wrap/reuse silently; choose a checked monotonic owner identifier or
another equally explicit collision-free-in-process contract during implementation review.

No unload or slot reuse means generation counters are unnecessary in this spec. Handles become invalid
when their renderer is destroyed. Resolve/validate all scene mesh/texture handles and material values
before acquire, mapped writes, fence reset, or recording any draw. An invalid object fails the entire
frame with ValidationError; preceding valid objects must not be partially submitted.

Mesh upload rejects empty vertices/indices, indices outside the supplied vertex slice, a triangle-list
index count not divisible by three, and checked-size/count conversion overflow. Validate before GPU
allocation. Existing finite/invertible-transform and normal-input requirements still apply. Do not
require every uploaded vertex to be referenced or invent geometry repair.

## Vulkan objects and ownership

An internal StoredMesh owns one device-local vertex buffer, one index buffer, and its index count.
The renderer owns an append-only mesh store. Upload reuses `Buffer::new_device_local` (the same
helper `buffer.rs` already uses for the single construction-time mesh today); there is no second
allocator or upload helper. Note: `Buffer::new_device_local`'s existing rollback (destroy the
just-created buffer on any copy failure) does not currently distinguish an ordinary copy failure from
a `VulkanError::Terminal` result (the GPU's completion could not be confirmed) — this is a pre-existing
gap in that helper, not something this spec introduces or is required to fix, but `StoredMesh`'s own
two-buffer upload inherits it exactly as the existing single-mesh path already does today.

Publish a handle only after both buffer uploads complete successfully. If the vertex buffer succeeds
and the index buffer fails, destroy the vertex buffer before returning the error (mirroring
`StoredTexture::new`'s existing pattern of releasing an already-created sibling resource on a later
step's failure). A failed upload must leave all existing handles valid and publish no partial store
entry.

**Validation/resolution must happen in a separate pass before any per-frame GPU work, not inline
during command recording.** Today, `resolve_texture_set` runs deep inside `FrameContext::record`
(after acquire, fence reset, and uniform/light updates) and tolerates an invalid handle by falling
back to white — that's why it's safe to call so late. Once invalid handles are rejected instead,
`render_objects_with_text` must validate every object's `MeshId` and `TextureId` against the
renderer's stores in one pass, in the same place and the same style as its existing camera/specular
validation loop (before checking `pending_resize`, before any acquire/fence/mapped-write), so an
invalid handle fails the whole frame before any of that happens. By the time `record`/
`emit_object_draws` runs, every object's handles are already known-valid; that stage resolves them
again (reusing the same resolution function, not a second implementation) but does not need to
return an error for an unresolvable handle, since validation already ruled that out.

For each object, resolve StoredMesh, bind its vertex/index buffers, bind its texture, push model and
material data from 013, then draw with that mesh's index count. This moves `cmd_bind_vertex_buffers`/
`cmd_bind_index_buffer` from their current single per-frame call (outside the object loop) into the
per-object path, and `emit_object_draws`/`ObjectCommand` gain a mesh-bind step and lose the single
externally-supplied `index_count` in favor of each object's own resolved mesh's count. The existing
`mixed_objects_bind_resolved_set_then_push_then_draw_in_order` test and its `record_objects` test
helper hardcode one shared index count for every object; both need rewriting to exercise interleaved
different-mesh objects, which is also exactly the Test plan's own "mesh A/B/A" requirement. Preserve
submission order. Pipeline, vertex format, camera set 0, texture set 1, and directional light set 2
remain shared across meshes.

Resources remain owned until renderer teardown; there is no early-destruction queue in this spec.
After completion, consume mesh buffers before allocator/device destruction. Buffer owners and native
handles stay private under 012. Creating a new mesh between frames must not destroy resources used by
older submissions. Retain the existing synchronous upload behavior; asynchronous uploads are deferred.

## Diagnostics and interactions

`scene_draw_calls` remains actual submitted scene draws. `submitted_indices` becomes the checked sum
of each submitted object's mesh index count, rather than objects.len() times one global count. Empty
frames report zero; failed acquire retains prior submitted counts. Text does not count as scene work.

Primary files: new `handle.rs` (shared owner-id newtype/counter) and `mesh.rs` (`MeshId`, `StoredMesh`,
the mesh store) modules; new `scene.rs` for `Object` (moved out of `texture.rs`, which keeps
`TextureId`/`SpecularMaterial`); `mod.rs`/`lib.rs` API and exports; `command.rs` resolution/binding/
counting; examples and README updated for the constructor/`render_frame` signature changes. No shader
changes are inherently required beyond the adopted 013 contracts.

## Test plan

- Validation: valid triangle/multi-triangle mesh, empty input, out-of-range indices, non-triangle index
  counts, and checked byte/count boundaries without allocating huge test arrays.
- Handle tests: correct owner/slot, foreign owner with same slot, invalid slot, distinct mesh/texture
  types, and existing handle stability after successful and failed uploads.
- Actual production draw-emission seam: interleaved mesh A/B/A uses the correct buffer pairs and
  index counts, preserves order, and sums diagnostics correctly. Empty input emits no scene draws.
- Failure injection at vertex/index upload and store publication; verify rollback/retention and no
  partial handle publication using 012's actual ownership backend.
- CPU CI and doctests must need no driver. Run all repository gates.

GPU acceptance: a cube and a differently sized triangle/plane in one frame, repeated use of both mesh
handles, textured/untextured objects, text-only frames with no meshes, creation between frames, resize,
and teardown under synchronization validation. Verify diagnostics against known differing index counts.

## Resolved decisions (2026-11-15)

- **Geometry-free construction + explicit MeshId:** accepted as proposed. Both existing constructors
  (`with_window`, `unsafe new`) lose their `vertices`/`indices` parameters; callers upload meshes via
  the new `create_mesh` and pass the returned `MeshId` through `Object`/`render_frame`. This is a
  breaking change to both constructors and to `render_frame`'s signature (gains a leading `mesh:
  MeshId` parameter) — migrate `examples/triangle.rs` and `examples/cube.rs` in the same change, same
  pattern as spec 013's `Camera` migration. `examples/window.rs` never constructs a `VulkanRenderer`
  (it's a winit-only sanity check) and needs no change.
- **TextureId strengthened to reject foreign/out-of-range handles:** accepted. `MeshId` gets strict
  rejection by necessity (there is no sensible silent default for a missing mesh — unlike a texture,
  which has an established white fallback). Extending the same rejection semantics to `TextureId`
  trades away its current silent-fallback-to-white behavior for consistency between the two handle
  types and for this project's established philosophy of rejecting invalid input rather than silently
  degrading it (same stance as `DirectionalLight`/`Camera`/`SpecularMaterial`/`PointLight` validation
  in specs 011/013/018). This requires rewriting the existing
  `resolve_none_and_out_of_range_fall_back_to_white` test in `texture.rs` — its name and assertions
  describe the behavior this spec deliberately removes; `None` (not a stale/foreign `Some`) remains
  the only way to request the white default.
- **Owner identifier mechanism and module placement:** a process-wide monotonically increasing
  counter (an `AtomicU64`, allocated once per `VulkanRenderer` construction) gives each renderer
  instance a distinct owner id. `MeshId`/`TextureId` each carry `(owner: u64, slot: usize)`; resolving
  either checks `owner` equality before indexing by `slot`, so a handle from one renderer instance can
  never resolve against another's store even if the slot numbers collide — satisfying the spec's
  "checked monotonic owner identifier" requirement. The counter and a small shared newtype live in a
  new `src/renderer/handle.rs` module (no existing module owns a concept like this); both the new mesh
  store and `texture.rs`'s existing `TextureId` consume it, so the identity check has exactly one
  implementation, not two. `TextureId` changes shape from today's bare `TextureId(pub(super) usize)`
  to carrying an owner id too — existing in-crate test code that constructs `TextureId(1)` directly
  (`command.rs`'s and `texture.rs`'s own test modules) will not compile unchanged; those call sites
  need a test-only helper that mints a validly-owned handle, not a bare tuple literal. `Object` moves
  out of `texture.rs` into a new `src/renderer/scene.rs`
  (Object now aggregates `MeshId` from the new mesh module alongside `TextureId`/`SpecularMaterial`
  from `texture.rs` — no single existing module owns all three, and `MeshId` is Object's primary
  field, so a dedicated scene-data module is cleaner than bolting it onto either resource module).

## Open questions for review

None blocking. Everything above was a design-blocking decision; all are resolved.

## Outcome

Implemented geometry-free windowed construction, explicit create_mesh/MeshId, per-object mesh
binding/counting, and strict renderer-owned mesh/texture resolution. Object now lives in scene.rs.
The original headless constructor remains a convenience fixture; new_empty/create_mesh support real
zero/multiple-mesh tests and render_frame_with_text exercises the shared text renderer. Invalid handles
are rejected before mapped writes/recording, leaving the next valid frame usable. Removed the temporary
MeshId::UNASSIGNED placeholder. Owner allocation now fails closed on u64 exhaustion without wrapping.

Added four opt-in GPU tests: different meshes, repeated handles, zero-mesh text, and invalid
handle/upload rejection followed by a valid render. Generated only three new multi_mesh_*.png files;
all previous baselines were left unchanged. The different-mesh/repeated-handle tests additionally
compare actual output with deliberately different draw inputs, so baseline generation cannot silently
accept the same geometry for every handle. Visually inspected all three new images.

Verification on the local Vulkan driver: 94 CPU library tests, 2 example tests, and doctests pass;
all 15 ignored GPU tests pass under synchronization validation without baseline-update mode. Existing
10 headless GPU tests also passed before new baseline generation. Formatting, warnings-denied Clippy,
and all-target build pass. A temporary seven-second windowed probe verified different mesh counts
(36 + 3 = 39 submitted indices, 2 scene draws), two resizes, minimize/restore, mesh creation between
frames, and normal shutdown with no emitted validation errors. The temporary source was removed.

Headless teardown retains its owner graph on terminal waits; text/upload entry points refuse reuse.
The pre-existing upload-helper terminal-cleanup limitation described above remains deferred under 012;
this change does not claim injected driver-failure coverage. Cross-platform runtime acceptance remains
future work. No mesh unloading, slot reuse, or GPU instancing is introduced.
