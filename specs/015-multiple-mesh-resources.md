# 015: Multiple mesh resources

**Status**: Draft for review; not implemented.
**Roadmap items**: Phase 3 resource handles; Phase 2 scenes containing different geometry.
**Dependencies**: Spec 012's private ownership/cleanup contract and spec 013's proposed Camera and
SpecularMaterial contracts. These interfaces are provisional until those drafts are reviewed.
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
pub struct MeshId { /* private renderer identity and slot */ }

pub struct Object {
    pub mesh: MeshId,
    pub model: Mat4,
    pub texture: Option<TextureId>,
    pub specular: SpecularMaterial,
}

impl VulkanRenderer {
    // Safety contract follows the final decision in 012; raw constructor shown here.
    pub unsafe fn new(
        display: RawDisplayHandle, window: RawWindowHandle, extent: (u32, u32),
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

Primary files: a private mesh owner/store module; Object's current namespace; mod.rs/lib.rs API and
exports; command.rs resolution/binding/counting; examples and README. Move Object out of texture.rs
only if needed for a clear shared scene-data namespace, updating exports rather than retaining duplicate
types. No shader changes are inherently required beyond the adopted 013 contracts.

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

## Open questions for review

- Accept geometry-free renderer construction and explicit MeshId in Object/render_frame?
- Accept strengthening TextureId to reject foreign/stale input in the same resource-identity change?
- Confirm the private owner identifier and module placement after 012's public boundary is settled.

## Outcome

Pending design review and implementation. No resource or rendering code changes in this draft.
