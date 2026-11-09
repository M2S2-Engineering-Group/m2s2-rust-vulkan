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

The minimum an engine needs to put anything on screen at all. This is implemented across `pipeline.rs`, `buffer.rs`, and
`command.rs`.

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
- [x] **Command pool & buffers**: recorded per-frame; structure recording so it can later be split per
      render pass (Phase 4 multi-threading) instead of one monolithic record function
- [x] **Synchronization**: semaphores (image-available, render-finished), fences (frame-in-flight), a named
      `FRAMES_IN_FLIGHT` constant rather than a hardcoded `2` scattered through the code
- [x] **Memory allocation strategy**: adopt `gpu-allocator` now rather than raw `vkAllocateMemory` per
      buffer/image — this is a foundational decision `buffer.rs` and later texture support both build on;
      switching allocators after buffer/image code exists is a rewrite, not an addition
- [x] **Vertex & index buffers**: staging buffer → device-local transfer pattern
- [x] **Swapchain recreation** on resize
- [x] **First triangle** rendered end-to-end

**Milestone**: hardcoded triangle renders and survives a window resize.

## Phase 2: Standard rendering features

What separates "a triangle renders" from "a usable rendering engine" for simple 3D scenes.

- [x] **Depth buffer** and depth testing — see [specs/006-uniforms-depth-camera.md](specs/006-uniforms-depth-camera.md)
- [x] **Uniform buffers & descriptor sets**: per-frame camera matrices (model transforms use push constants) via `m2s2-math` — see [specs/006](specs/006-uniforms-depth-camera.md)
- [x] **Push constants** for small, frequently-updated per-draw data — [spec 007](specs/007-multiple-objects-and-push-constants.md)
- [x] **Camera**: view/projection using `src/math.rs`'s existing `perspective`/`look_at` — see [specs/006](specs/006-uniforms-depth-camera.md)
- [x] **Textures**: image loading, `VkImage`/`VkImageView` creation, layout transitions, samplers — [spec 010](specs/010-textures.md). Mipmaps deferred to a follow-up spec.
- [x] **Multiple draw calls**: independently transformed copies of one mesh through one pipeline — [spec 007](specs/007-multiple-objects-and-push-constants.md)
- [ ] **Basic instancing**: deferred; 007 issues separate indexed draws
- [x] **Text drawing and diagnostic overlay**: reusable text submissions; camera position, FPS/CPU frame interval, scene counts, device and swapchain state — [spec 008](specs/008-diagnostic-text-overlay.md). Shared sampled-R8 uploads are available; general textures remain below.
- [ ] **Basic lighting**: Phong/Blinn-Phong (ambient + diffuse + specular), normals, one or more point/
      directional lights. _Partially done_ — [spec 011](specs/011-directional-lighting.md) implemented
      per-vertex normals and a single directional ("sun") light with per-pixel diffuse + ambient
      shading (plus the cube example's free-fly camera), and
      [spec 013](specs/013-specular-lighting.md) added the Blinn-Phong specular highlight (per-object
      color/shininess, camera-dependent) on top of it. Ambient + diffuse + specular + normals are now
      done; point/spot lights and multiple lights remain — box stays unchecked until those land.

**Milestone**: a lit, textured 3D scene with a movable camera and more than one object.

**Safety cleanup (done):** [spec 012](specs/012-resource-ownership-and-memory-safety.md)
landed the resource-ownership/memory-safety work ahead of further rendering features — private owners +
safe `with_window`/`unsafe new` constructors, bounded exclusive mapped writes, ownership-consuming
destruction, and a terminal-failure leak policy. Two scoped items deferred (ManuallyDrop→Option for the
six core owners; upload staging leak-on-terminal) — see the spec Outcome.

Proposed follow-ups after cleanup: [draft spec 013](specs/013-specular-lighting.md) adds per-object
Blinn–Phong highlights and explicit frame camera data; [draft spec 014](specs/014-texture-mipmaps.md)
adds optional generated texture mip chains. Both are awaiting review; neither is implemented.

Additional drafts (all unimplemented):

- [015: Multiple mesh resources](specs/015-multiple-mesh-resources.md) — typed mesh handles and renderer identity; builds on 012/013.
- [016: GPU instancing](specs/016-gpu-instancing.md) — consecutive mesh/texture batches; builds on 015 and per-object material data.
- [017: Directional shadow mapping](specs/017-directional-shadow-mapping.md) — explicit two-pass shadows; pass interfaces provisional, with 016 reused when present.
- [018: Point lighting](specs/018-point-lighting.md) — bounded positioned lights; builds on 011–013 and can precede shadow implementation.

Numbering identifies documents, not a mandatory execution sequence. Resolve ownership contracts first;
point lighting and mipmaps can proceed independently of the mesh/instancing/shadow sequence.

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

- [x] **Headless rendering and golden-image test infrastructure**: `#[cfg(test)]`-only offscreen
      rendering (no window/surface) plus PNG-backed golden-image comparison with tolerance, so any
      future rendering feature can be tested for actual pixel correctness, not just "it compiles and
      doesn't crash." Pulled forward as a prerequisite for spec 018 (point lighting)'s GPU acceptance
      tests — see spec 019.
- [ ] **Debug tooling integration**: `VK_EXT_debug_utils` object naming and command buffer labels, so
      RenderDoc/validation output is legible per-consumer
- [ ] **Bindless / descriptor indexing**: reduces per-draw descriptor churn, relevant once Phase 3's asset
      count grows
- [ ] **Robust resize/minimize handling**: zero-size swapchains, device-lost recovery
- [ ] **Performance profiling hooks**: timestamp queries, GPU frame timing exposed to the caller
- [ ] **Ray tracing** (stretch): `VK_KHR_ray_tracing_pipeline` / ray query, only if a consumer needs it

Items here can be pulled forward if a specific milestone in the tutorial project or the C++ port needs them
sooner — this phase is a backlog, not a strict sequence.
