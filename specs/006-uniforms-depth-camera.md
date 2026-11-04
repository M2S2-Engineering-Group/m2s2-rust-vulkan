# 006: First 3D object — uniform buffers, descriptor sets, depth, camera

**Roadmap item**: [ROADMAP.md](../ROADMAP.md) Phase 2 — **Depth buffer and depth testing**, **Uniform
buffers & descriptor sets** (per-frame MVP via `m2s2-math`), and **Camera** (view/projection using
`src/math.rs`). Three checkboxes, one coherent "first real 3D" milestone.

## Goal

Render the caller-supplied geometry transformed by a model/view/projection matrix, depth-tested, so a
spinning 3D **cube** draws with correct occlusion. This is the step from "a flat quad renders" to "a 3D
object renders under a camera," and it introduces the two concepts the rest of Phase 2 builds on: a
**depth attachment** in the dynamic-rendering flow, and **uniform buffers + descriptor sets** as the way
per-frame data reaches shaders.

The per-frame transform is driven **by the caller**, not the library. This is deliberate and matches the
no-event-loop architecture (see [CLAUDE.md](../CLAUDE.md) Architecture): the consumer's game/update layer
runs movement/physics/collision, produces each object's **world-space model matrix**, and hands that plus
the camera's view/projection to the renderer, which does the `proj * view * model` combine on the GPU. The
simulation side never needs to know the camera exists — that is the update/render separation this API is
shaped around.

## Non-goals

Deferred to later Phase 2 specs / roadmap items:

- **Multiple objects / instancing** — a single object only. The API grows into N objects later via a
  per-frame UBO for the camera (view+proj) plus a per-object model matrix fed through **push constants**
  (itself a Phase 2 item); this spec does not build that.
- **Push constants** — the model matrix goes in the UBO here, not a push constant.
- **Textures, samplers, mipmaps** — no image resources.
- **Lighting / normals** — the cube is flat-shaded with per-vertex color; keeping `model` separate in the
  UBO is the only concession to the later lighting work.
- **Depth-only / shadow passes** — the depth buffer is a single attachment on the main pass, not a
  reusable multi-pass resource (Phase 4).

## Public API

The one breaking change is `render_frame`, which now takes the transform. Everything else is new, crate-
internal (`pub(crate)`) plumbing.

```rust
// mod.rs — VulkanRenderer
// Was: pub fn render_frame(&mut self) -> Result<()>
pub fn render_frame(&mut self, model: Mat4, view: Mat4, proj: Mat4) -> Result<()>;

// uniform.rs (new) — owns UBOs + descriptor machinery, FRAMES_IN_FLIGHT of each
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

pub(crate) struct Uniforms {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    buffers: [Buffer; FRAMES_IN_FLIGHT], // host-visible, persistently mapped
}

impl Uniforms {
    pub(crate) fn new(device: &VulkanDevice, allocator: &mut Allocator) -> Result<Self>;
    /// Write `ubo` into frame `frame`'s mapped uniform buffer. No GPU sync needed: the buffer is
    /// per-frame-in-flight, and `draw_frame` has already waited on that frame's fence.
    pub(crate) fn update(&self, frame: usize, ubo: &UniformBufferObject);
    pub(crate) fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout;
    pub(crate) fn descriptor_set(&self, frame: usize) -> vk::DescriptorSet;
    /// Destroys pool + layout via `device`, frees the mapped buffers via `allocator`.
    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator);
}

// depth.rs (new) — swapchain-sized depth attachment, recreated on resize
pub(crate) struct DepthResources {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    format: vk::Format,
}

impl DepthResources {
    pub(crate) fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<Self>;
    pub(crate) fn view(&self) -> vk::ImageView;
    pub(crate) fn format(&self) -> vk::Format;
    pub(crate) fn destroy(&mut self, device: &VulkanDevice, allocator: &mut Allocator);
}

/// Chooses the depth format: prefers D32_SFLOAT, else the first supported of
/// {D32_SFLOAT_S8_UINT, D24_UNORM_S8_UINT} with OPTIMAL-tiling depth-stencil-attachment support.
/// Pure given the queried format properties, so it is unit-testable without a device.
fn select_depth_format(candidates: &[(vk::Format, vk::FormatFeatureFlags)]) -> Option<vk::Format>;

// buffer.rs — EXTEND the existing Buffer (shared-before-one-off), do not add a parallel type.
impl Buffer {
    /// A HOST_VISIBLE | HOST_COHERENT buffer kept persistently mapped, for CPU-updated-per-frame data
    /// (UBOs). Returns the buffer; the mapped pointer is reached through the retained `Allocation`.
    pub(crate) fn new_host_visible_mapped(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self>;
}
```

`GraphicsPipeline::new` also changes shape (see Interactions): it gains the descriptor-set-layout and the
depth format, so the pipeline layout references the set and the pipeline enables depth test/write.

## Vulkan objects

New objects this spec creates / owns / destroys:

- **Depth attachment** (`DepthResources`): one `vk::Image` (`D32_SFLOAT` preferred, `OPTIMAL` tiling,
  `DEPTH_STENCIL_ATTACHMENT` usage), its `vk::ImageView` (depth aspect), and a device-local gpu-allocator
  `Allocation`. Sized to the swapchain extent; **recreated whenever the swapchain is** (resize).
- **Uniform buffers** (`Uniforms`): `FRAMES_IN_FLIGHT` host-visible, persistently-mapped `vk::Buffer`s +
  allocations, each sized `size_of::<UniformBufferObject>()`.
- **Descriptor set layout**: one binding — binding 0, `UNIFORM_BUFFER`, `VERTEX` stage, count 1.
- **Descriptor pool**: capacity for `FRAMES_IN_FLIGHT` uniform-buffer descriptors / sets. Its sets are
  freed implicitly when the pool is destroyed — no `free_descriptor_sets`.
- **Descriptor sets**: `FRAMES_IN_FLIGHT`, each `vkUpdateDescriptorSets`-pointed at its own uniform buffer.

**Ownership.** Both `DepthResources` and `Uniforms` are owned by **`VulkanRenderer`**, not by
`VulkanSwapchain`. This is forced by the existing allocator rule: freeing a gpu-allocator `Allocation`
needs `&mut Allocator`, which a `Drop` impl on `VulkanSwapchain` does not have — the same reason `Buffer`
is not `Drop` and is torn down explicitly (see [specs/003](003-vertex-buffers.md)). So these are destroyed
explicitly in `VulkanRenderer::Drop`, not via field-drop.

**Destroy order** (extends the chain in `mod.rs`'s `Drop`, after `wait_idle`):

```
frame_context
  -> pipeline                 (its pipeline layout references the descriptor-set-layout; destroy pipeline first)
  -> uniforms.destroy()       (descriptor pool + set layout via device; the N mapped buffers via allocator)
  -> depth.destroy()          (image + view via device; allocation via allocator)
  -> vertex_buffer / index_buffer .destroy()   (via allocator)
  -> allocator                (ManuallyDrop::drop — must outlive every allocation freed above)
  -> swapchain
  -> device
  -> instance
```

The descriptor-set-layout must be **created before the pipeline**, because the pipeline layout consumes it
at pipeline-creation time. Creation order in `VulkanRenderer::new`: allocator → vertex/index buffers →
`DepthResources` → `Uniforms` (creates the set layout) → `GraphicsPipeline` (takes the set layout + depth
format) → `FrameContext`.

## Interactions

- **`pipeline.rs`** — `GraphicsPipeline::new` gains two params: the descriptor-set-layout (for the
  `vk::PipelineLayoutCreateInfo::set_layouts`) and the depth `vk::Format` (added to
  `vk::PipelineRenderingCreateInfo::depth_attachment_format`). It enables depth via
  `vk::PipelineDepthStencilStateCreateInfo` (`depth_test_enable`, `depth_write_enable`, compare op
  `LESS`).
- **`command.rs`** — `record` adds a depth `vk::RenderingAttachmentInfo` (load `CLEAR` to depth 1.0, store
  `DONT_CARE`) to `vk::RenderingInfo::depth_attachment`, and binds the per-frame descriptor set
  (`cmd_bind_descriptor_sets`, `GRAPHICS`, pipeline layout, set index 0) before the draw. A layout
  transition for the depth image to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` is added alongside the existing
  color barrier (`UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, early/late-fragment-tests stages).
  `draw_frame` gains the three matrices from `render_frame`, assembles a `UniformBufferObject`, and calls
  `uniforms.update(frame, &ubo)` before submit. The depth image comes from the renderer (passed down like
  the swapchain is).
- **`mod.rs`** — `render_frame` signature change; it builds the `UniformBufferObject` pass-through. The
  resize branch (from [specs/004](004-swapchain-recreation.md)) recreates `DepthResources` to the new
  extent right after `swapchain.recreate` (device already idle there). `Uniforms` is extent-independent
  and untouched on resize.
- **`buffer.rs`** — gains `new_host_visible_mapped` next to `new_device_local`; both share the existing
  allocation/creation path where possible.
- **`shaders/`** — `triangle.vert` gains a `layout(set=0, binding=0) uniform UBO { mat4 model, view,
  proj; }` and outputs `proj * view * model * vec4(inPosition, 1.0)`. Recompiled to
  `triangle.vert.spv` via `glslc` and committed (the offline-SPIR-V pattern from [specs/001](001-graphics-pipeline.md)).
- **`examples/triangle.rs`** (consumer, owns the geometry) — replace the quad with an 8-vertex / 36-index
  **cube** (per-vertex color; winding consistent with the pipeline's clockwise front face + back-face
  cull). Switch `ControlFlow::Wait -> Poll` so it animates. Each frame compute `model` (spin: `rotate`
  about Y by elapsed time), `view` (`look_at` from a pulled-back eye), `proj` (`perspective`, aspect from
  the current window size) and pass them to `render_frame`.

## Test plan

- **Unit (CI, no GPU — mirrors `device.rs`/`swapchain.rs`/`error.rs`):**
  - `select_depth_format` — returns `D32_SFLOAT` when it advertises the depth-stencil feature; falls back
    to the first supported stencil combo; returns `None` when nothing qualifies.
  - `size_of::<UniformBufferObject>()` equals `3 * 64` (three `mat4`s, no padding) — guards the std140
    assumption and catches an accidental field-type change.
  - Cube geometry sanity in the example-adjacent pure helper if one exists (36 indices); otherwise asserted
    in the example.
- **Manual:** `cargo run --example triangle` shows a solid, correctly-occluded **spinning cube** (far
  faces hidden behind near ones — the depth test working). Resizing the window keeps it correct (depth
  recreated to the new extent). Validation layers (spec 005) stay silent throughout.

## Open questions

Resolve during implementation; record the answer in Outcome.

1. **Vulkan Y-flip.** `perspective_rh_zo` targets Vulkan's [0,1] depth, but Vulkan's clip-space Y points
   down vs. OpenGL's up. May need `proj[1][1] *= -1` (or equivalent) or the cube renders vertically
   mirrored. Confirm against what `m2s2-math`'s `perspective_rh_zo` already does before adding a flip.
2. **Matrix memory order.** `m2s2-math`'s `Matrix4x4f32` storage (row- vs column-major) vs. GLSL's
   column-major default. If they differ, either transpose on upload or declare `layout(row_major)` on the
   UBO block. Determine empirically (a correct vs. transposed cube is visually obvious).
3. **`Mat4: bytemuck::Pod`.** Assumed to hold under `m2s2-math`'s `bytemuck` feature, as it does for
   `Vec3` (used in `Vertex`). If not, store `[[f32; 4]; 4]` in `UniformBufferObject` and convert on
   `update`.
4. **Depth load/store + layout.** `DONT_CARE` store is assumed (depth isn't read after the pass). Confirm
   no validation complaint about the initial `UNDEFINED` depth layout with a `CLEAR` load.

## Outcome

Implemented as specified; the module split (`depth.rs`, `uniform.rs`, extended `buffer.rs`), ownership,
and destroy order all held up. Pure logic was TDD'd (`select_depth_format`, 3 tests, red→green); the UBO
layout has a guard test. 23 unit tests pass; fmt/clippy/build all clean.

**Open questions resolved** (all four confirmed against `m2s2-math`'s actual source):

1. **Y-flip — needed.** `perspective_rh_zo` produces an OpenGL-style Y (positive `f` in row 1, no flip),
   so without correction the scene is vertically mirrored. Fixed in `math::perspective` by negating
   element `(1,1)` — keeping the Vulkan convention in that layer, which already documented it. This had a
   **knock-on**: the Y-flip inverts screen-space winding, so the pipeline's front face was changed
   `CLOCKWISE → COUNTER_CLOCKWISE` (pipeline.rs) and the cube authored CCW-outward. (The old CLOCKWISE
   suited the pre-projection clip-space quad.)
2. **Matrix order — row-major.** `m2s2-math` stores `data[row * cols + col]`; GLSL defaults to
   column-major, so the shader's UBO block is declared `layout(row_major)`. Size (192) is unaffected.
3. **`Mat4: Pod` — holds** under the `bytemuck` feature; the `[[f32;4];4]` fallback wasn't needed.
4. **Depth load/store/layout** — `DONT_CARE` store and an `UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
   transition every frame are validation-clean.

**Deviation from the spec's signatures:** `DepthResources::new` also takes `&VulkanInstance` (not just
`&VulkanDevice`) — the depth-format query needs `get_physical_device_format_properties`, which lives on the
instance. `VulkanDevice` exposes `physical_device` but no instance handle.

**Verification.** Ran `cargo run --example triangle` on an Apple M5 (kosmickrisp Vulkan driver) with
validation layers ON: it created the pipeline, ran the render loop over many frames, and handled a window
resize (exercising depth-image recreation) with **zero validation errors/warnings**. Visual confirmation of
the spinning cube (orientation, occlusion, colors) is left to a human with the window on screen — a headless
session can't observe rendered pixels, only that the GPU work is validation-clean. If the cube looks
mirrored or inside-out on a given setup, the `(1,1)` flip and the front-face winding are the two knobs.

### Follow-up correctness review

The pre-007 review corrected shared-depth synchronization, depth replacement on allocation failure,
and initial depth sizing from the selected swapchain extent. See [REVIEW_NOTES.md](../REVIEW_NOTES.md)
for those fixes, the related initialization audit, runtime findings, and verification limits.
