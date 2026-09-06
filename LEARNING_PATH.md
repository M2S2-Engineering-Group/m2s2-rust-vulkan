# Learning Path

Education plan for building `m2s2-rust-vulkan`, structured to track [ROADMAP.md](ROADMAP.md) phase by phase.
Each roadmap phase pairs with the Vulkan/Rust/graphics concepts it requires, so studying stays a step ahead
of implementation instead of catching up after the fact.

This supersedes the previous version of this file, which referenced `nalgebra` and a `window.rs` module that
no longer exist in this codebase (math is `m2s2-math`; window ownership was deliberately moved out of the
library — see [CLAUDE.md](CLAUDE.md)'s Architecture section).

## How to use this

For each roadmap item you're about to implement:
1. Read the "Concepts to learn" for that phase below, and the linked resource, *before* writing the spec
   (see [SPEC_PROCESS.md](SPEC_PROCESS.md)).
2. Write the spec. Writing down the Vulkan objects and API surface is itself a comprehension check — if you
   can't state what objects you need and their lifetimes, you're not ready to implement yet.
3. Implement against the spec.
4. Update the checkbox in `ROADMAP.md` and note anything that surprised you (a wrong assumption, a Vulkan
   validation error that taught you something) in the spec file itself, not a separate notes doc.

## Prerequisites

- [x] Rust fundamentals (ownership, borrowing, `unsafe`) — already in active use throughout `src/renderer/`
- [ ] Linear algebra: vectors, matrices, transformations — covered by `m2s2-math`; know what it gives you
      before Phase 2 needs MVP matrices
- [x] Vulkan SDK installed with validation layers (required for debug-build validation, see
      [CLAUDE.md](CLAUDE.md))

## Phase 0 (done) — what to understand retroactively

The existing code already implements this; if you inherited or skimmed past it, close the gap before Phase 1:
- Instance/device/queue model, why `device.rs` scores physical devices instead of taking the first one
- Why surface creation is centralized in `VulkanInstance::create_surface` (`instance.rs`) instead of
  duplicated per-caller
- The swapchain destroy-order bug `swapchain.rs` fixed
  (`VUID-vkDestroySurfaceKHR-surface-01266`) — read the Architecture section in `CLAUDE.md` and the Vulkan
  spec page for that VUID; this is the single most instructive real bug in the codebase so far
- **Resource**: [Vulkan Tutorial](https://vulkan-tutorial.com/) chapters 1–3 (Instance, Physical/Logical
  devices, Presentation) map directly to `instance.rs`/`device.rs`/`swapchain.rs`

## Phase 1: Core render loop

**Concepts to learn**:
- Render pass models: classic `VkRenderPass`/`VkFramebuffer` vs. `VK_KHR_dynamic_rendering` — read both
  before implementing `pipeline.rs`, since ROADMAP.md Phase 1 commits to dynamic rendering; understand *why*
  it's preferred here, not just how to call it
- Graphics pipeline state: vertex input, input assembly, rasterization, color blending — these become
  `VkGraphicsPipelineCreateInfo` fields
- SPIR-V and the shader compilation model (GLSL/HLSL → SPIR-V via `shaderc`)
- Command buffers: primary vs. secondary, one-time-submit vs. reusable
- Synchronization: semaphores (GPU-GPU), fences (CPU-GPU), why both are needed for one frame
- GPU memory types and allocation strategy — why a suballocator (`gpu-allocator`) exists instead of one
  `VkDeviceMemory` per buffer (there's a driver-enforced minimum allocation count on many GPUs)
- Staging buffers: why device-local memory usually isn't host-visible, and the copy pattern that follows

**Resources**:
- [Vulkan Tutorial](https://vulkan-tutorial.com/) chapters 4–15 (render pass through vertex buffers) — the
  closest 1:1 match to this phase, written for the render-pass model; cross-reference the
  [dynamic rendering extension docs](https://docs.vulkan.org/samples/latest/samples/extensions/dynamic_rendering/README.html)
  for the version this repo actually uses
- [`ash` documentation](https://docs.rs/ash/latest/ash/) — primary API reference for translating tutorial
  C++ to this codebase's Rust bindings
- [`gpu-allocator` docs](https://docs.rs/gpu-allocator/latest/gpu_allocator/) — read before writing
  `buffer.rs`, not after

**Milestone check**: matches ROADMAP.md Phase 1 — triangle renders, survives resize.

## Phase 2: Standard rendering features

**Concepts to learn**:
- Depth testing and the depth buffer as an attachment
- Descriptor sets/layouts and uniform buffer objects — how shader-visible data actually gets to the GPU
- Push constants vs. UBOs — when each is the right tool
- View/projection matrix conventions — read `src/math.rs`'s doc comments on why it calls `perspective_rh_zo`/
  `look_at_rh` specifically (Vulkan's clip-space and depth-range conventions differ from OpenGL's)
- Image layout transitions, image views, samplers, mip chains
- Phong/Blinn-Phong lighting model: ambient/diffuse/specular, normal vectors in view vs. world space

**Resources**:
- [Vulkan Tutorial](https://vulkan-tutorial.com/) chapters 16–24 (uniform buffers through mipmapping)
- [Learn OpenGL](https://learnopengl.com/) — lighting chapters translate directly; OpenGL's simpler API
  surface makes the lighting math easier to isolate from Vulkan boilerplate while learning it

**Milestone check**: matches ROADMAP.md Phase 2 — lit, textured scene, movable camera, multiple objects.

## Phase 3: Scene & asset infrastructure

**Concepts to learn**:
- glTF format basics (scenes, nodes, meshes, materials, accessors) — what a loader actually has to parse
- Handle-based resource management patterns — why engines don't hand out raw API objects to callers
- Frustum culling: view-frustum plane extraction from a projection matrix, AABB/sphere tests

**Resources**:
- [glTF 2.0 spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) or the `gltf` crate's docs for
  a Rust-side loader instead of hand-rolling the parser
- [Real-Time Rendering](https://www.realtimerendering.com/) — culling and scene-representation chapters

**Milestone check**: matches ROADMAP.md Phase 3 — external glTF model renders through the resource-handle
API.

## Phase 4: Advanced rendering

**Concepts to learn**:
- Render graphs: pass dependency resolution, attachment aliasing
- Shadow mapping: depth-only render, projective sampling, common artifacts (peter-panning, acne) and their
  standard fixes (slope-scaled bias, normal offset)
- Deferred shading: G-buffer layout tradeoffs, why it front-loads geometry cost to defer lighting cost
- PBR: metallic/roughness workflow, the rendering equation at a level sufficient to implement Cook-Torrance
- Compute shaders: workgroups, shared memory, barriers — distinct execution model from the graphics pipeline

**Resources**:
- [Real-Time Rendering](https://www.realtimerendering.com/) — the standard reference for all of the above
- [GPU Gems](https://developer.nvidia.com/gpugems/gpugems/contributors) — shadow mapping and deferred
  shading chapters specifically
- [LearnOpenGL PBR](https://learnopengl.com/PBR/Theory) — theory translates directly; implementation doesn't
  (descriptor sets vs. uniforms), which is itself a useful exercise in seeing where Vulkan and OpenGL diverge

**Milestone check**: matches ROADMAP.md Phase 4 — deferred, shadowed, PBR-lit scene with post-processing.

## Phase 5: Engine-level concerns

**Concepts to learn**:
- `VK_EXT_debug_utils`: object naming, command buffer labels, and how they show up in RenderDoc
- Descriptor indexing / bindless design tradeoffs
- Swapchain edge cases: zero-size windows (minimized), `VK_ERROR_DEVICE_LOST` recovery
- GPU timestamp queries for frame timing

**Resources**:
- [RenderDoc](https://renderdoc.org/) — install it now, use it from Phase 1 onward for frame debugging, not
  just when Phase 5 formalizes the integration
- [Vulkan Configurator (vkconfig)](https://vulkan.lunarg.com/doc/sdk/latest/windows/vkconfig.html) —
  validation layer configuration

**Milestone check**: matches ROADMAP.md Phase 5 — this phase is a backlog; pull items forward as the tutorial
project or C++ port need them.

## Tips

1. Get each roadmap milestone actually working before starting the next phase's spec — this codebase has
   already been broken by skipping this once (`cargo build --examples` silently drifted from the winit API;
   see `CLAUDE.md`'s Status section).
2. Run with validation layers on (`cfg(debug_assertions)`, already wired up in `instance.rs`) at all times
   during development.
3. A spec you can't finish writing is a sign you don't understand the feature yet — that's the spec process
   doing its job, not a reason to skip it.
