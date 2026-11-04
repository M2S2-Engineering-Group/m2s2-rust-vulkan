# 007: Multiple objects and push constants

**Status**: Implemented; visual acceptance checks remain as noted in Outcome.
**Roadmap items**: [Phase 2](../ROADMAP.md#phase-2-standard-rendering-features), push constants and multiple draw calls.
**Prerequisite**: [Correctness review](../REVIEW_NOTES.md); automated GPU resize/minimize/shutdown checks were exercised during 007.

## Goal

Draw two independently transformed copies of the caller's uploaded mesh using one pipeline and one
camera. Separate per-frame camera data from per-draw model transforms.

## Non-goals

Multiple meshes, GPU instancing, textures, lighting, scene ownership, resource handles, and render graphs.
The caller continues to own animation, camera behavior, window creation, and the event loop.

## Public API

Keep `VulkanRenderer::new` unchanged: one vertex/index pair is uploaded and owned by the renderer.
Add this method (all transforms refer to that same mesh):

```rust
pub fn render_objects(&mut self, models: &[Mat4], view: Mat4, proj: Mat4) -> Result<()>;
```

Keep `render_frame(model, view, proj)` as a compatibility wrapper that calls `render_objects` with
`std::slice::from_ref(&model)`. Share the resize/acquire/submit/present implementation rather than
creating a second frame loop. An empty slice still clears and presents a frame with no geometry.

Internal shader data types:

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniforms {
    pub view: Mat4,
    pub proj: Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ModelPushConstants {
    pub model: Mat4,
}
```

Use named constants for descriptor binding, push-constant offset (zero), size (derived from the Rust
type), and stage visibility (`VERTEX`). Keep the camera at set 0, binding 0. Declare `row_major` in
both GLSL blocks to match `m2s2-math` storage. The vertex shader computes
`camera.proj * camera.view * object.model * vec4(inPosition, 1.0)`.

## Vulkan objects

No additional objects per draw. Existing per-frame uniform buffers shrink from three matrices to two.
The pipeline layout adds one vertex-stage push-constant range containing a single 64-byte model matrix.
Validate the requested range against `maxPushConstantsSize` before pipeline creation, returning a clear
initialization error if unsupported. Do not introduce a fallback data path.

Before the loop, bind the pipeline, camera descriptor set, vertex buffer, and index buffer once.
For each model, record `vkCmdPushConstants` followed by `vkCmdDrawIndexed` with instance count one.
Push-constant bytes are copied into recorded command state; the input slice needs no GPU lifetime.

Ownership and teardown remain unchanged: wait for device idle, destroy frame resources and pipeline,
destroy uniforms/depth/mesh buffers, then allocator, swapchain, device, and instance. The shared depth
attachment keeps the explicit dependency introduced by the correctness review.

## Interactions

- `uniform.rs`: replace the MVP block with camera-only data; retain per-frame fence-protected updates.
- `pipeline.rs`: declare the push-constant range and validate device limits.
- `command.rs`: accept the model slice and record one push/draw pair per element.
- `mod.rs`: expose `render_objects`, with `render_frame` forwarding to it.
- `shaders/triangle.vert`: update both blocks; rebuild and commit its SPIR-V using the existing `glslc` workflow.
- `examples/cube.rs`: animate two cubes with different positions/rotations, including overlap to demonstrate depth.
- `examples/triangle.rs`: retain the existing single-object call to exercise compatibility.

## Test plan

CPU-only tests:
- Camera/model payload sizes and field offsets match the GLSL layout (128 and 64 bytes).
- Push-constant range covers the complete payload at offset zero and only the vertex stage.
- Device-limit validation accepts equal/larger limits and rejects smaller limits.
- Shader SPIR-V still parses after recompilation.
- Test command emission through a small recording test seam: an empty slice emits no push/draw,
  one model emits one pair, and two different models emit two pairs with the correct bytes/order.
  Use the same recording logic in production; do not test a separate simulated draw implementation.

Manual GPU checks with synchronization validation enabled:
- Two cubes move independently with one camera and one pipeline.
- Overlap occludes correctly regardless of submission order.
- An empty slice displays only the clear color; the triangle compatibility path still renders.
- Repeated resize, minimize/restore, and normal shutdown produce no validation errors.

Run all four repository quality gates. Only check off push constants and multiple draw calls in the
roadmap after implementation and verification; instancing remains deferred.

## Open questions

None required for this scope. Multiple distinct meshes will need a later resource API spec; this method
explicitly draws repeated copies of the existing mesh.

## Outcome

Implemented as specified: camera UBOs are 128 bytes, model push constants are 64 bytes, and
`render_frame` forwards to `render_objects`. The cube example draws two independently animated copies
of one uploaded mesh. The vertex SPIR-V was rebuilt with `glslc` and checked with `spirv-val`.

The model payload and range live in `pipeline.rs`; `VulkanDevice` caches the queried push-constant size
limit so pipeline creation can validate it without changing its public constructor. Command emission
uses one callback path shared by production and tests; it allocates no intermediate command list.

All 36 CPU tests and the repository quality gates pass. A synchronization-validation run of the cube
emitted no errors. A temporary automated example also exercised two resizes, minimize/restore, empty
frames, the single-object wrapper, and event-loop exit with normal destruction; it exited successfully
without validation errors. The temporary source was removed afterward.

Visual readability of both cubes and depth occlusion independent of draw order still require human
inspection; the smoke checks validate API execution/synchronization, not rendered pixels. GPU instancing
remains deferred. The proposed diagnostic text overlay is captured separately in draft spec 008.
