# 003: Vertex and index buffers

**Roadmap item**: [ROADMAP.md](../ROADMAP.md) Phase 1 — "Memory allocation strategy" and "Vertex & index
buffers".

**Depends on**: [001-graphics-pipeline.md](001-graphics-pipeline.md) (the pipeline's vertex-input state) and
[002-command-buffers-and-render-loop.md](002-command-buffers-and-render-loop.md) (the draw recording).

**Learning path**: [LEARNING_PATH.md](../LEARNING_PATH.md) Phase 1 — vertex/staging/index buffer sections.
Vulkan Tutorial's [Vertex buffers](https://vulkan-tutorial.com/Vertex_buffers/Vertex_buffer_creation),
[Staging buffer](https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer), and
[Index buffer](https://vulkan-tutorial.com/Vertex_buffers/Index_buffer) chapters — **but** we use
`gpu-allocator` instead of the tutorial's manual `vkAllocateMemory` (see Decision).

## Goal

Replace spec 001's hardcoded `gl_VertexIndex` positions with real per-vertex data fed from a GPU **vertex
buffer**, plus an **index buffer**, so `cargo run --example triangle` draws a **quad** (two indexed
triangles, four differently-colored corners). Introduce `gpu-allocator` as the memory backend, wrapped
behind a `Buffer` type so nothing else in the codebase touches the allocator directly.

## Non-goals

- **Writing our own allocator.** Deliberately using `gpu-allocator` now. Building a Level-1 sub-allocator by
  hand — behind the same `Buffer` interface, as a drop-in comparison — is a planned later *learning spike*,
  not this spec (see Decision for why that split).
- **Uniform buffers / descriptor sets / MVP matrices.** Vertices pass straight through in clip space
  (`z = 0`); no transforms yet — Phase 2.
- **Dynamic geometry.** The quad is static: uploaded once at startup, never updated. No per-frame re-upload,
  no persistently-mapped vertex buffer.
- **A mesh/asset system.** One hardcoded quad, defined in code. A `Mesh`/`Geometry` abstraction and file
  loading are much later.
- **Depth / 3D.** `vec3` positions are used (`z = 0`) so the vertex format is 3D-ready, but there is no depth
  buffer or perspective yet (Phase 2).

## Decision: `gpu-allocator`, wrapped — own allocator deferred

Raw Vulkan memory is one `vkAllocateMemory` + `vkBindBufferMemory` per resource, with manual memory-type
selection. Two problems make that a dead end at any real scale: the driver's `maxMemoryAllocationCount`
limit (often ~4096 total), and the need to sub-allocate offsets within a few big blocks (with alignment and
`bufferImageGranularity` rules). A correct sub-allocator is a real, bug-prone systems project, and a bug
there is silent memory aliasing — especially dangerous while `VK_LAYER_KHRONOS_validation` is still off (see
001/002 Outcomes).

So: adopt **`gpu-allocator`** (pure-Rust, battle-tested) now, and **wrap it behind our own `Buffer`/`Allocator`
types** (Dependency Inversion — the rest of the crate depends on our abstraction, never on `gpu-allocator`
directly). Because memory management *is* core Vulkan learning (unlike, say, FFI bindings), a hand-written
Level-1 sub-allocator behind that same interface is a worthwhile **future spike** — kept out of this spec so
it doesn't block the vertex-buffer milestone or entangle a half-built allocator into first buffer code.

## Decision: staging buffer → device-local

Vertex/index data lives in `DEVICE_LOCAL` memory (fastest for the per-frame GPU reads) and is uploaded via a
temporary `HOST_VISIBLE` **staging buffer** + a one-time `vkCmdCopyBuffer`. The simpler alternative (a
host-visible vertex buffer, no staging) is left unused: the staging→device-local transfer is needed for all
static geometry anyway, so establishing it now is the right foundation rather than a shortcut to redo.

## Public API

```rust
// src/renderer/buffer.rs

use m2s2_math::Vector3f32 as Vec3; // #[repr(C)] over [f32; 3]; Pod via m2s2-math's `bytemuck` feature

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3, // 12 bytes @ offset 0
    pub color: Vec3,    // 12 bytes @ offset 12
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription;        // binding 0, stride = size_of::<Vertex>(), VERTEX
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2]; // loc 0 pos @ offset 0, loc 1 color @ offset 12, both R32G32B32_SFLOAT
}

pub type Index = u16; // vk::IndexType::UINT16 at bind time

/// A DEVICE_LOCAL GPU buffer, memory supplied by the wrapped allocator. Not `Drop` — freeing a
/// gpu-allocator allocation needs `&mut Allocator`, so teardown is the explicit `destroy` below,
/// called before the Allocator is dropped (see Vulkan objects / Open questions).
pub struct Buffer {
    pub(crate) buffer: vk::Buffer, // read by command.rs to bind
    allocation: Option<gpu_allocator::vulkan::Allocation>, // Option so destroy() can take it to free
    len: u32, // element count, for cmd_draw_indexed / records
}

impl Buffer {
    /// Create a DEVICE_LOCAL buffer with `usage | TRANSFER_DST`, uploaded from `data` through a
    /// transient HOST_VISIBLE staging buffer + a one-time copy submitted on the graphics queue.
    pub(crate) fn new_device_local<T: bytemuck::Pod>(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self>;

    pub(crate) fn len(&self) -> u32;
    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator);
}
```

```rust
// src/renderer/allocator.rs  (new module — thin wrapper so the dependency stays localized)

pub struct Allocator { /* wraps gpu_allocator::vulkan::Allocator; Drop frees its device-memory blocks */ }

impl Allocator {
    pub fn new(instance: &VulkanInstance, device: &VulkanDevice) -> Result<Self>;
    // pub(crate) allocate/free helpers used by Buffer
}
```

The quad is 4 `Vertex` (corners at `±0.5`, `z = 0`, distinct colors) and 6 `Index` (`[0, 1, 2, 2, 3, 0]`),
defined in `buffer.rs`. If `Vec3::new` is not `const`, the vertex data is a small constructor `fn` rather
than a `const` array (`[f32; 3]` would allow a `const` — minor cost of using `Vec3`). Winding must match the
pipeline's `front_face = CLOCKWISE` / `cull = BACK` from 001 — if the quad renders black/empty, reversed
winding is the first suspect (same gotcha 001 flagged).

**Why `Vec3`, not `[f32; 3]`** (decided during review): position is genuinely a point in space and shares the
type the Phase-2 matrices will transform; `m2s2-math` is built for this (its `bytemuck` feature + `repr(C)`
exist precisely so its vectors can go into GPU buffers). Color as `Vec3` is the conventional graphics choice
and keeps a single vector type across the crate; a dedicated `Color` type is a reasonable future refinement
(see Open questions). The `size_of::<Vertex>() == 24` unit test guards the layout regardless.

## Vulkan objects

| Object | Count | Created by | Lifetime |
|---|---|---|---|
| `Allocator` (owns the backing `VkDeviceMemory` blocks) | 1 | `Allocator::new` | Owned by `VulkanRenderer`; dropped **after** every `Buffer` is freed and **before** the `VkDevice` |
| vertex `Buffer` (`VkBuffer` + sub-allocation) | 1 | `Buffer::new_device_local`, `DEVICE_LOCAL`, usage `VERTEX_BUFFER | TRANSFER_DST` | Owned by `VulkanRenderer`; freed to the allocator before it's dropped |
| index `Buffer` | 1 | same, usage `INDEX_BUFFER | TRANSFER_DST` | same |
| staging buffer | per upload | inside `new_device_local`, `HOST_VISIBLE | HOST_COHERENT`, `TRANSFER_SRC` | Freed before `new_device_local` returns — never retained |
| transient command pool + buffer (the copy) | per upload | inside `new_device_local` | Destroyed before `new_device_local` returns; copy is submitted + waited on a fence |

**Destroy order** (extends 002's chain): `frame_context → pipeline → index_buffer → vertex_buffer →
allocator → swapchain → device → instance`. Buffers are freed before the allocator; the allocator before the
device. The buffers are plain fields with an explicit `destroy(&device, &mut allocator)` call in
`VulkanRenderer::Drop` (not `ManuallyDrop`), because freeing needs `&mut Allocator` — `Drop` alone can't reach
it. `allocator`, like the others, is `ManuallyDrop` and torn down in that explicit block.

## Interactions

- **`pipeline.rs` (001):** vertex-input state stops being empty — `GraphicsPipeline::new` now uses
  `Vertex::binding_description()` + `Vertex::attribute_descriptions()`. No signature change (the format is
  static); it just imports `buffer::Vertex`.
- **`command.rs` (002):** `draw_frame` records `cmd_bind_vertex_buffers`, `cmd_bind_index_buffer` (UINT16),
  and `cmd_draw_indexed(index_count, …)` instead of `cmd_draw(3, …)`. **Signature change:** `draw_frame` takes
  the vertex and index `&Buffer` (passed from `render_frame`).
- **`mod.rs`:** `VulkanRenderer` gains `allocator`, `vertex_buffer`, `index_buffer` fields; constructs them
  after the swapchain (allocator, then the two buffers) and before the pipeline/frame context; `render_frame`
  passes the buffers to `draw_frame`; `Drop` extends the teardown chain + the explicit buffer `destroy` calls.
- **shaders:** `triangle.vert` takes `layout(location=0) in vec3 inPosition; layout(location=1) in vec3
  inColor;` instead of the hardcoded arrays (recompile + recommit `triangle.vert.spv`). `triangle.frag`
  unchanged. The shaders become geometry-agnostic pass-throughs (names kept — see Open questions).
- **`error.rs`:** add a `VulkanError::AllocationError(String)` variant for `gpu-allocator` failures (a distinct
  failure domain — Types Over Strings). (Open question: or reuse `InitializationError`.)
- **`Cargo.toml`:** add `gpu-allocator` (version pinned at implementation time); enable `m2s2-math`'s
  `bytemuck` feature (`m2s2-math = { version = "0.2", features = ["bytemuck"] }`) so `Vec3` is `Pod`.
- **`src/math.rs`:** re-export `Vec3` (`Vector3f32`) if it doesn't already, so `buffer.rs` and future code
  share one alias.

## Test plan

- **Unit tests** (pure logic, no live Vulkan — more is testable here than in 001/002):
  - `size_of::<Vertex>() == 24`.
  - `Vertex::binding_description()`: `stride == size_of::<Vertex>()`, `input_rate == VERTEX`, `binding == 0`.
  - `Vertex::attribute_descriptions()`: position `location 0`, `offset 0`, `R32G32B32_SFLOAT`; color
    `location 1`, `offset 12`, `R32G32B32_SFLOAT`.
- **Manual verification** (this spec's milestone): `cargo run --example triangle` shows a **quad** (its four
  corners red/green/blue/yellow, smoothly interpolated across the face), larger than the old triangle. Close
  the window; confirm a clean shutdown with no validation errors — the teardown chain now has the allocator +
  two buffers freed in the right order.

## Open questions

- **Buffer-free ergonomics.** `gpu-allocator` requires `&mut Allocator` to free an allocation, so `Buffer`
  can't free itself in `Drop`. Chosen: explicit `destroy(&device, &mut allocator)` in `VulkanRenderer::Drop`
  (consistent with the codebase's explicit `ManuallyDrop` teardown). Alternative: `Arc<Mutex<Allocator>>` held
  by each `Buffer` for RAII `Drop` — heavier, and overkill for two static buffers. Revisit if many buffers
  make manual teardown unwieldy.
- **Error variant.** New `AllocationError(String)` vs. reusing `InitializationError`. Leaning new variant
  (named failure domain), but it's a one-line judgment call.
- **`draw_frame` signature growth.** Passing two `&Buffer` now; a `Mesh { vertex, index, count }` bundle will
  be nicer once there's more than one object to draw (Phase 2 "multiple draw calls"). Deferring the
  abstraction until there's a second caller.
- **Dedicated `Color` type.** Color is a `Vec3` for now (conventional, one vector type). A real `Color` (or
  `Vec4` with alpha) type would be semantically truer — fair future refinement, not worth a new type for one
  quad today.
- **Shader filenames.** Keep `triangle.{vert,frag}` (now generic pass-throughs) or rename to `mesh`/`geometry`.
  Keeping them avoids churn; flagging in case a rename reads better.

## Outcome

_(Filled in after implementation.)_
