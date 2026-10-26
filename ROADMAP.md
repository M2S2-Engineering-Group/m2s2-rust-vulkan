# Engine Roadmap

Feature roadmap for `m2s2-rust-vulkan`, organized the way most real-time rendering engines factor their
feature set: a render loop, standard rendering features, scene/asset infrastructure, then advanced rendering.
Each phase lists what a typical engine has at that stage and what it maps to in this repo.

This file answers "what are we building and in what order." For "how do I learn what I need to build it," see
[LEARNING_PATH.md](LEARNING_PATH.md). For "how a feature gets specified before it's coded," see
[SPEC_PROCESS.md](SPEC_PROCESS.md).

Status of each phase's items is tracked as checkboxes here; broader done/not-done narrative for what already
exists lives in [CLAUDE.md](CLAUDE.md#status) — this file doesn't duplicate that, only extends it forward.

## Phase 0: Foundations — done

Instance/device/swapchain, validation layers, error handling, the `m2s2-math` layer, CI quality gates, and
release automation. See [CLAUDE.md](CLAUDE.md#status) for the detailed list. Nothing left to do here to move
on to Phase 1.

## Phase 1: Core render loop

The minimum an engine needs to put anything on screen at all. This is what `pipeline.rs`, `buffer.rs`, and
`command.rs` are stubbed out for.

- [x] **Shader pipeline**: GLSL compiled offline to SPIR-V via `glslc`, committed to `shaders/*.spv`, loaded
      via `include_bytes!` + `ash::util::read_spv` — see
      [specs/001-graphics-pipeline.md](specs/001-graphics-pipeline.md) for why build-time `shaderc`
      compilation was passed over for now
- [x] **Graphics pipeline object**: vertex input, input assembly, viewport/scissor, rasterization state,
      color blend state
- [x] **Render target strategy**: dynamic rendering (`VK_KHR_dynamic_rendering`, core in Vulkan 1.3) vs.
      classic render pass + framebuffer objects — **decide this before writing `pipeline.rs`**, not after.
      Dynamic rendering avoids framebuffer-object bookkeeping and composes far better with the multi-pass
      work in Phase 4 (shadow maps, deferred shading, post-processing all become "render into a different
      set of attachments," not "build another framebuffer graph"). Retrofitting this later means rewriting
      `pipeline.rs` and every draw call.
- [ ] **Command pool & buffers**: recorded per-frame; structure recording so it can later be split per
      render pass (Phase 4 multi-threading) instead of one monolithic record function
- [ ] **Synchronization**: semaphores (image-available, render-finished), fences (frame-in-flight), a named
      `FRAMES_IN_FLIGHT` constant rather than a hardcoded `2` scattered through the code
- [ ] **Memory allocation strategy**: adopt `gpu-allocator` now rather than raw `vkAllocateMemory` per
      buffer/image — this is a foundational decision `buffer.rs` and later texture support both build on;
      switching allocators after buffer/image code exists is a rewrite, not an addition
- [ ] **Vertex & index buffers**: staging buffer → device-local transfer pattern
- [ ] **Swapchain recreation** on resize (the `triangle.rs` resize handler is currently a `TODO`)
- [ ] **First triangle** rendered end-to-end

**Milestone**: hardcoded triangle renders and survives a window resize.

## Phase 2: Standard rendering features

What separates "a triangle renders" from "a usable rendering engine" for simple 3D scenes.

- [ ] **Depth buffer** and depth testing
- [ ] **Uniform buffers & descriptor sets**: per-frame MVP matrices via `m2s2-math`
- [ ] **Push constants** for small, frequently-updated per-draw data
- [ ] **Camera**: view/projection using `src/math.rs`'s existing `perspective`/`look_at`
- [ ] **Textures**: image loading, `VkImage`/`VkImageView` creation, layout transitions, samplers, mipmaps
- [ ] **Multiple draw calls / basic instancing**: more than one object per frame without one pipeline per
      object
- [ ] **Basic lighting**: Phong/Blinn-Phong (ambient + diffuse + specular), normals, one or more point/
      directional lights

**Milestone**: a lit, textured 3D scene with a movable camera and more than one object.

## Phase 3: Scene & asset infrastructure

Since this library is explicitly scoped to Vulkan mechanics (see [CLAUDE.md](CLAUDE.md)'s Architecture
section — no window/event-loop ownership), scene graph and asset ownership need a clear boundary: what the
library owns vs. what it hands back to the caller (the tutorial project and the C++ port each want scene
management their own way).

- [ ] **Resource handles**: opaque handles for GPU resources (buffers, images, pipelines) instead of
      exposing raw `vk::Buffer`/`vk::Image` to callers — this is the point where "library" vs. "app code"
      boundaries get decided
- [ ] **Model loading**: glTF (preferred — PBR-friendly material model) and/or OBJ
- [ ] **Material system**: decouple "what a shader needs" from "what an asset provides"
- [ ] **Scene graph**: hierarchical transforms — likely belongs to the *caller*, not this library, given the
      no-ownership architecture; decide explicitly and document it rather than let it default
- [ ] **Frustum culling**: basic visibility culling before draw submission

**Milestone**: load and render an external glTF model with materials, through the resource-handle API rather
than raw Vulkan handles.

## Phase 4: Advanced rendering

This is the phase the Phase 1 decisions (dynamic rendering, per-pass-splittable command recording,
`gpu-allocator`) are specifically there to make cheap instead of expensive.

- [ ] **Render graph / multi-pass architecture**: ordered passes with attachment dependencies, built on
      dynamic rendering from Phase 1
- [ ] **Shadow mapping**: depth-only pass + sampling
- [ ] **Deferred shading**: G-buffer pass + lighting pass
- [ ] **Post-processing**: tone mapping, bloom, screen-space effects as additional passes in the render graph
- [ ] **PBR materials**: metallic/roughness workflow, image-based lighting
- [ ] **MSAA** / other anti-aliasing
- [ ] **Compute shaders**: GPGPU pass type, particle systems
- [ ] **Multi-threaded command recording**: split across the per-pass command buffers from Phase 1

**Milestone**: a deferred-shaded, shadowed, PBR-lit scene with post-processing.

## Phase 5: Engine-level concerns

Cross-cutting concerns that matter because this is a *reusable library*, not a single application.

- [ ] **Debug tooling integration**: `VK_EXT_debug_utils` object naming and command buffer labels, so
      RenderDoc/validation output is legible per-consumer
- [ ] **Bindless / descriptor indexing**: reduces per-draw descriptor churn, relevant once Phase 3's asset
      count grows
- [ ] **Robust resize/minimize handling**: zero-size swapchains, device-lost recovery
- [ ] **Performance profiling hooks**: timestamp queries, GPU frame timing exposed to the caller
- [ ] **Ray tracing** (stretch): `VK_KHR_ray_tracing_pipeline` / ray query, only if a consumer needs it

Items here can be pulled forward if a specific milestone in the tutorial project or the C++ port needs them
sooner — this phase is a backlog, not a strict sequence.
