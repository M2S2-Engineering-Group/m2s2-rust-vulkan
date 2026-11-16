// Caller-facing mesh handles and the renderer-owned mesh store. See spec 015 (multiple mesh
// resources). Wired into `mod.rs` (`create_mesh`, the renderer's `meshes` store) and `command.rs`
// (per-object mesh resolution/binding) as of Task 4.

use super::{
    allocator::Allocator,
    buffer::{Buffer, Index, Vertex},
    device::VulkanDevice,
    handle,
};
use crate::error::{Result, VulkanError};
use ash::vk;

/// Opaque handle to a renderer-owned mesh: an `(owner, slot)` pair identifying one entry in the
/// renderer's mesh store. `Copy`; valid for the lifetime of the renderer that produced it. Fields
/// are private to this module (not even `pub(super)`) — nothing outside `mesh.rs`, including the
/// renderer itself, can construct a `MeshId` directly; `MeshId::new` (below) is the sole
/// constructor, called only from `mod.rs`'s `create_mesh`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MeshId {
    owner: u64,
    slot: usize,
}

impl MeshId {
    /// Construct a `MeshId` for a renderer's `owner` id and a slot in its mesh store. Mirrors
    /// `texture::TextureId::new` — the sole constructor, called from `mod.rs`'s `create_mesh` (the
    /// production path) and from this module's own tests.
    pub(super) fn new(owner: u64, slot: usize) -> Self {
        Self { owner, slot }
    }
}

/// Checked `usize -> u32` conversion for a vertex/index count. GPU buffer lengths (`Buffer::len`)
/// and draw-call counts are `u32`; this rejects a count that wouldn't survive that conversion
/// instead of silently truncating it.
fn checked_count(len: usize) -> Result<u32> {
    u32::try_from(len)
        .map_err(|_| VulkanError::ValidationError("mesh vertex/index count exceeds u32".into()))
}

/// Validate caller-supplied mesh geometry before any GPU work: neither `vertices` nor `indices` may
/// be empty, `indices.len()` must be a multiple of 3 (whole triangles only), every index must
/// address a real vertex (an index equal to `vertices.len()` is one past the end and is rejected,
/// not silently clamped), and both counts must survive the `u32` conversion GPU buffer lengths and
/// draw calls use.
pub(crate) fn validate_mesh_input(vertices: &[Vertex], indices: &[Index]) -> Result<()> {
    if vertices.is_empty() {
        return Err(VulkanError::ValidationError("mesh has no vertices".into()));
    }
    if indices.is_empty() {
        return Err(VulkanError::ValidationError("mesh has no indices".into()));
    }
    if !indices.len().is_multiple_of(3) {
        return Err(VulkanError::ValidationError(
            "mesh index count is not a multiple of 3".into(),
        ));
    }
    for &index in indices {
        if index as usize >= vertices.len() {
            return Err(VulkanError::ValidationError(format!(
                "mesh index {index} is out of range for {} vertices",
                vertices.len()
            )));
        }
    }
    checked_count(vertices.len())?;
    checked_count(indices.len())?;
    Ok(())
}

/// One renderer-owned mesh: a vertex buffer, an index buffer, and the index count for
/// `cmd_draw_indexed`. Not `Drop`: freeing needs `&mut Allocator` — call `destroy` explicitly
/// before the `Allocator` is torn down (mirrors `buffer::Buffer`/`texture::StoredTexture`). Fields
/// are `pub(super)` (not just private) because Task 4's command recording, a sibling module of
/// `mesh`, needs the raw buffers and index count to bind/draw.
pub(crate) struct StoredMesh {
    pub(super) vertex_buffer: Buffer,
    pub(super) index_buffer: Buffer,
    pub(super) index_count: u32,
}

impl StoredMesh {
    /// Destroy both buffers and return their allocations to `allocator`. Must run before the
    /// `Allocator` is torn down. Signature/consuming-`self` shape mirrors
    /// `texture::StoredTexture::destroy`.
    pub(super) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        self.vertex_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
    }
}

/// Upload `vertices`/`indices` into a device-local vertex buffer and a device-local index buffer.
/// Validates the input first (`validate_mesh_input`) so no GPU work happens on malformed geometry.
/// Mirrors `texture::StoredTexture::new`'s two-resource rollback: if the index buffer upload fails,
/// the vertex buffer is destroyed before returning the error.
pub(crate) fn upload_mesh(
    device: &VulkanDevice,
    allocator: &mut Allocator,
    vertices: &[Vertex],
    indices: &[Index],
) -> Result<StoredMesh> {
    validate_mesh_input(vertices, indices)?;
    let vertex_buffer = Buffer::new_device_local(
        device,
        allocator,
        vertices,
        vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let index_buffer = match Buffer::new_device_local(
        device,
        allocator,
        indices,
        vk::BufferUsageFlags::INDEX_BUFFER,
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            vertex_buffer.destroy(device, allocator);
            return Err(error);
        }
    };
    // `validate_mesh_input` above already confirmed `indices.len()` fits a `u32`.
    let index_count = indices.len() as u32;
    Ok(StoredMesh {
        vertex_buffer,
        index_buffer,
        index_count,
    })
}

/// Resolve a `MeshId` against `owner` (the renderer instance's own owner id) and `store` (its mesh
/// store), using the shared `handle::resolve_handle` identity check. Returns `ValidationError` — not
/// a silent fallback — on a mismatched owner or an out-of-range slot; there is no sensible default
/// mesh.
pub(crate) fn resolve_mesh(id: MeshId, owner: u64, store: &[StoredMesh]) -> Result<&StoredMesh> {
    match handle::resolve_handle(id.owner, id.slot, owner, store.len()) {
        Some(slot) => Ok(&store[slot]),
        None => Err(VulkanError::ValidationError(
            "mesh id does not belong to this renderer or is out of range".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Vec2, Vec3};

    fn vertex() -> Vertex {
        Vertex {
            position: Vec3::new(0.0, 0.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            uv: Vec2::new(0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
        }
    }

    #[test]
    fn validate_rejects_empty_vertices() {
        let indices: [Index; 3] = [0, 0, 0];
        assert!(validate_mesh_input(&[], &indices).is_err());
    }

    #[test]
    fn validate_rejects_empty_indices() {
        let vertices = [vertex(), vertex(), vertex()];
        assert!(validate_mesh_input(&vertices, &[]).is_err());
    }

    #[test]
    fn validate_rejects_index_one_past_the_end() {
        let vertices = [vertex(), vertex(), vertex()];
        // Valid index range is 0..=2 (three vertices); 3 is one past the end.
        let indices: [Index; 3] = [0, 1, 3];
        assert!(validate_mesh_input(&vertices, &indices).is_err());
    }

    #[test]
    fn validate_rejects_index_count_not_multiple_of_three() {
        let vertices = [vertex(), vertex(), vertex(), vertex()];
        let indices: [Index; 4] = [0, 1, 2, 3];
        assert!(validate_mesh_input(&vertices, &indices).is_err());
    }

    #[test]
    fn validate_accepts_valid_multi_triangle_mesh() {
        let vertices = [vertex(), vertex(), vertex(), vertex()];
        let indices: [Index; 6] = [0, 1, 2, 2, 3, 0];
        assert!(validate_mesh_input(&vertices, &indices).is_ok());
    }

    #[test]
    fn checked_count_rejects_values_that_overflow_u32() {
        // Smallest input that overflows the u32 conversion, tested against the helper directly with
        // plain integers rather than allocating a multi-gigabyte array.
        assert!(checked_count(u32::MAX as usize).is_ok());
        assert!(checked_count(u32::MAX as usize + 1).is_err());
    }

    // resolve_mesh identity-check tests (Step 5). Task 1's handle.rs tests already cover
    // resolve_handle (the shared primitive) in isolation; these two confirm mesh.rs wires it
    // correctly for MeshId specifically, using an empty store so no real GPU-backed StoredMesh is
    // needed (owner mismatch and out-of-range-slot both fail before any indexing happens). The third
    // case this step asks for -- "matching owner + valid slot resolves to the right StoredMesh" --
    // needs an actual StoredMesh to prove it returns the *right* one, which needs a real
    // VulkanDevice/Allocator. That full round trip is below as a GPU-gated integration test
    // (`#[ignore = "needs a real Vulkan driver"]`, the same convention `headless.rs` already uses),
    // not faked here.
    #[test]
    fn resolve_mismatched_owner_is_rejected() {
        let owner = handle::next_owner_id();
        let other = handle::next_owner_id();
        let store: [StoredMesh; 0] = [];
        let id = MeshId { owner, slot: 0 };
        assert!(resolve_mesh(id, other, &store).is_err());
    }

    #[test]
    fn resolve_out_of_range_slot_is_rejected() {
        let owner = handle::next_owner_id();
        let store: [StoredMesh; 0] = [];
        let id = MeshId { owner, slot: 0 };
        assert!(resolve_mesh(id, owner, &store).is_err());
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn upload_and_resolve_round_trip_on_a_real_device() {
        use super::super::allocator::Allocator;
        use super::super::device::VulkanDevice;
        use super::super::instance::VulkanInstance;

        let instance = VulkanInstance::new_headless().expect("instance");
        let device = VulkanDevice::new_headless(&instance).expect("device");
        let mut allocator = Allocator::new(&instance, &device).expect("allocator");

        let vertices = [vertex(), vertex(), vertex()];
        let indices: [Index; 3] = [0, 1, 2];
        let stored = upload_mesh(&device, &mut allocator, &vertices, &indices).expect("upload");
        assert_eq!(stored.index_count, 3);
        let store = [stored];

        let owner = handle::next_owner_id();
        let id = MeshId { owner, slot: 0 };

        // Matching owner + valid slot resolves to the one mesh in the store.
        let resolved = resolve_mesh(id, owner, &store).expect("resolves");
        assert_eq!(resolved.index_count, 3);

        // Mismatched owner and out-of-range slot are both rejected against the same real store.
        let other = handle::next_owner_id();
        assert!(resolve_mesh(id, other, &store).is_err());
        let out_of_range = MeshId { owner, slot: 1 };
        assert!(resolve_mesh(out_of_range, owner, &store).is_err());

        let [stored] = store;
        stored.destroy(&device, &mut allocator);
    }
}
