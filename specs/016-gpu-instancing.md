# 016: GPU instancing

**Status**: Draft for review; not implemented.
**Roadmap item**: Phase 2 basic instancing.
**Dependencies**: 012's exclusive per-frame writes, 013's Camera/SpecularMaterial, and 015's MeshId,
Object and validated resource store. Depends on those design contracts; no implementation prerequisite
is needed to review this draft. Mipmaps and shadows are independent.

## Goal

Render consecutive objects sharing one mesh and texture with a single indexed instanced draw. Keep
per-object transforms and specular materials independent and preserve input order.

## Non-goals

Global sorting, indirect draws, bindless textures, compute culling, a scene graph, transparency sorting,
per-instance descriptor indexing, animation/skinning, or a second legacy rendering path.

## Public API and behavior

No new draw API. Accept the Object slices from 015 and compile them into consecutive compatible runs.
The batch key is resolved MeshId plus resolved texture identity. Material/model differences are instance
data and do not split a batch. None resolves to the white default; foreign handles remain validation
errors under 015. Never merge nonconsecutive runs: A/B/A stays three draws.

Use one instanced scene pipeline for singleton and multi-object runs; instanceCount=1 is valid. Remove
the scene model/material push-constant path once its data moves to the instance stream. Do not maintain
two independent implementations of the same object rendering concept.

Proposed fixed capacity: `MAX_INSTANCES_PER_FRAME = 4096` objects across the complete frame, including
singleton runs. Validate the whole frame before acquire/writes/fence reset. Empty input clears/presents;
over-capacity input returns ValidationError, never truncates or partially draws. Check all usize/u32
conversions and diagnostics arithmetic. Camera/material/resource validation from earlier specs remains.

This draft proposes adding `scene_instances: usize` to RendererStats and formatting it in diagnostics.
`scene_draw_calls` reports actual batches; `submitted_indices` sums mesh indexCount * instanceCount.
Counts update only after submission and exclude text and any later shadow pass.

## GPU data and Vulkan objects

A fixed-size mapped instance vertex buffer per frame slot, owned by the renderer and written only
after that slot's fence signals. Payload stride is 80 bytes:
- Four model rows: offsets 0/16/32/48, each vec4.
- Specular color.xyz and shininess.w: one vec4 at offset 64.

Keep mesh vertex binding 0 and locations 0–3 from 011. Instance binding 1 uses inputRate INSTANCE;
model rows occupy locations 4–7 and material data location 8, all R32G32B32A32_SFLOAT. Validate the
binding count, attribute count/offsets, and stride against device limits before creating the pipeline.
No vertex-divisor extension, descriptor indexing, or new optional device feature is required.

GLSL matrix constructors take columns: reconstruct the row-major model with
`transpose(mat4(row0, row1, row2, row3))`. Transform positions/normals exactly as 013, including the
inverse-transpose model normal matrix. Pass specular data to the fragment shader using flat varyings.
Camera/light UBOs and texture descriptor bindings remain unchanged.

Build instance data once in original object order, using a reusable bounded CPU vector. Bind the full
per-frame instance buffer at offset zero. Each batch records mesh/texture binding and
`vkCmdDrawIndexed(indexCount, batch_length, 0, 0, first_instance)`; first_instance is the batch's offset
in that buffer. Do not apply both a buffer-offset adjustment and firstInstance for the same range.

Remove the scene push range and declarations consistently from both scene shaders/pipeline creation.
Text still has its independent vertex stream and pipeline. Construction rollback, resize, teardown,
and terminal frame-error handling follow 012. The instance buffer need not resize with the window.

## Batching algorithm

Validate/resolve all objects first. Walk in input order; extend the current run only when its resolved
mesh/texture key matches the next object. Each run records its first instance, length, and mesh count.
Use the same plan for upload order, draw recording, and counters. Do not build separate simulated
batching logic only for tests. Reuse mesh/texture resolution from 015.

All shaders, input layout, and instance data must change in one coherent implementation unit. Tests
must not accept repeated ordinary draws as an implementation of this spec.

## Demonstration and test plan

CPU:
- Instance size/offsets and explicit matrix row reconstruction with a nonsymmetric transform; verify
  material payload and shader SPIR-V input decorations, not just a matrix identity case.
- Empty, singleton, all-compatible, alternating, and A/A/B/B/A inputs. Material differences do not
  split; texture/mesh differences do. Check firstInstance offsets and counts.
- Capacity at/below/above 4096 and checked representation boundaries; no partial frame on rejection.
- Recorded draws come from the production batching/emission seam; verify instanceCount, firstInstance,
  buffer offsets, mesh counts, and diagnostics. Multiple frame slots never share the writable buffer.
- GPU-allocation failure/teardown and mapped-capacity tests reuse 012's implementation test backend.

GPU: many independently transformed copies with different shininess, several mesh/texture runs, and
non-uniform scales. Compare actual pixels against the unbatched pre-016 baseline with a recorded
tolerance and use a frame capture to establish that one compatible run is one instanced draw. Validate
text after the instanced scene, empty frames, resize/minimize, and shutdown. A CPU batching test alone
cannot establish correct instance attribute addressing or matrix reconstruction.

Run all repository gates and doctests; compile/validate changed SPIR-V. Keep GPU tests opt-in.

## Open questions for review

- Accept contiguous batching as the only initial policy and 4096 as the explicit frame budget?
- Confirm attribute locations and RendererStats.scene_instances before API migration.
- Agree on the comparison fixture/tolerance and a demonstrable draw-count reduction target.

## Outcome

Pending review and implementation. Interface details follow the final accepted 013/015 contracts.

## References

- [Indexed draw instanceCount/firstInstance](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndexed.html)
- [Vertex binding input rates and limits](https://docs.vulkan.org/refpages/latest/refpages/source/VkVertexInputBindingDescription.html)
