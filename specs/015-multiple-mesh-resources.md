# 015: Multiple mesh resources

**Status**: Draft for review; not implemented.
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
The renderer owns an append-only mesh store. Upload uses the shared staging/submission path and the
pending-resource retention/error rules from 012; there is no second allocator or upload helper.

Publish a handle only after both uploads complete successfully. Failure of the second upload must
release completed earlier resources when safe and retain any still-pending resources as 012 requires.
A failed upload must leave all existing handles valid and publish no partial store entry.

For each object, resolve StoredMesh, bind its vertex/index buffers, bind its texture, push model and
material data from 013, then draw with that mesh's index count. Preserve submission order. Pipeline,
vertex format, camera set 0, texture set 1, and directional light set 2 remain shared across meshes.

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
  MeshId` parameter) — migrate `examples/{triangle,cube,window}.rs` in the same change, same pattern
  as spec 013's `Camera` migration.
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
  implementation, not two. `Object` moves out of `texture.rs` into a new `src/renderer/scene.rs`
  (Object now aggregates `MeshId` from the new mesh module alongside `TextureId`/`SpecularMaterial`
  from `texture.rs` — no single existing module owns all three, and `MeshId` is Object's primary
  field, so a dedicated scene-data module is cleaner than bolting it onto either resource module).

## Open questions for review

None blocking. Everything above was a design-blocking decision; all are resolved.

## Outcome

Pending implementation.
