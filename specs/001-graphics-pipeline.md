# 001: Graphics pipeline creation

**Roadmap item**: [ROADMAP.md](../ROADMAP.md) Phase 1 — "Shader pipeline", "Graphics pipeline object", and
"Render target strategy" bullets.

**Learning path**: [LEARNING_PATH.md](../LEARNING_PATH.md) Phase 1. Read the dynamic rendering vs. render
pass material there before continuing — this spec assumes you've seen both models, not just the one this repo
picks.

## Goal

Replace the `pipeline.rs` stub with a `GraphicsPipeline` that can be created and destroyed cleanly (zero
validation errors) against the existing `VulkanDevice`/`VulkanSwapchain`, using `VK_KHR_dynamic_rendering`
instead of `VkRenderPass`/`VkFramebuffer`. Shader source is hardcoded vertex positions/colors in the vertex
shader — no vertex buffer input yet.

## Non-goals

- **Vertex/index buffers.** The vertex shader hardcodes three positions via `gl_VertexIndex` (the same
  approach vulkan-tutorial.com uses before it introduces vertex buffers). Pipeline's vertex input state is
  therefore empty (no bindings, no attributes). Real buffers, replacing the hardcoded positions, are
  `003-vertex-buffers.md` — deliberately *after* command buffers (`002`), since hardcoding the positions in
  the shader means we don't need real buffers to reach the "first triangle" milestone.
- **Command buffer recording / actually drawing anything.** This spec produces a pipeline object, not pixels
  on screen. Recording `vkCmdBeginRendering` → `vkCmdBindPipeline` → `vkCmdDraw` → `vkCmdEndRendering`,
  including the manual image layout transitions dynamic rendering requires (no render pass to do them for
  you), is `002-command-buffers-and-render-loop.md`.
- **Wiring into `VulkanRenderer`.** `GraphicsPipeline` stays standalone (constructed directly from
  `&VulkanDevice` + a color format) until `002` needs to drive it from the render loop. Avoids touching
  `mod.rs` before there's a reason to.
- **Descriptor sets, push constants, depth testing.** No uniform data crosses into the shader yet — Phase 2.
  Pipeline layout is empty.

## Decision: dynamic rendering over render pass objects

Confirming the call `ROADMAP.md` already made, with the reasoning spelled out: `VK_KHR_dynamic_rendering`
lets `vkCmdBeginRendering` take attachments directly instead of requiring a `VkRenderPass` + matching
`VkFramebuffer` built ahead of time. For Phase 4 (shadow maps, deferred G-buffer passes, post-processing),
each is "begin rendering into a different attachment set" rather than "construct another framebuffer object
graph." The cost is paid once, now, instead of as a rewrite when Phase 4 arrives.

**Consequence for `device.rs`** (required before this spec's code compiles against a real driver — flagging
as a dependency, not doing it silently): `VulkanDevice::new` must additionally:
1. Add `ash::khr::dynamic_rendering::NAME` to the required device extensions. `supports_required_extensions`
   currently hardcodes a single name (`VK_KHR_swapchain`) — this should become a check over a small
   `const REQUIRED_DEVICE_EXTENSIONS: &[&CStr]` array instead of a second hardcoded `.any(...)` call (Types
   Over Strings: a second bespoke check next to the first is exactly the "parallel implementation of the same
   concept" `CLAUDE.md` says to avoid).
2. Chain `vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true)` into
   `DeviceCreateInfo` via `.push_next(...)`.

The instance API version can stay at `VK_API_VERSION_1_0` — this uses the extension, not the Vulkan 1.3 core
promotion, so it works on any driver that exposes the extension rather than requiring a full 1.3 driver.

## Decision: precompiled SPIR-V, not build-time `shaderc`

`ROADMAP.md` currently says "build-time, via shaderc" for the shader pipeline bullet. Revising that here: the
`shaderc` crate's `shaderc-sys` build script compiles glslang/shaderc from source (needs cmake + a C++
toolchain) unless prebuilt binaries are available for the CI runner's exact target, which adds real build
time and a real "does this even compile on `ubuntu-latest`" risk to a CI pipeline that CLAUDE.md already notes
has no Vulkan SDK or GPU. For a single hardcoded triangle shader pair, that cost isn't earned yet.

Instead: `.vert`/`.frag` GLSL source lives in `shaders/`, compiled offline with `glslc` (part of the Vulkan
SDK, already a prerequisite per `README.md`) into `shaders/*.spv`, committed to the repo, and loaded at
compile time via `include_bytes!`. No build.rs, no new build dependency, CI keeps building without a shader
compiler present. Revisit build-time compilation if/when shader hot-reload during development becomes worth
the added complexity — that's a later-phase problem, not this one.

*(Housekeeping: I'll update `ROADMAP.md`'s Phase 1 bullet to match this decision so the docs don't
immediately drift from what this spec commits to.)*

## Public API

```rust
// src/renderer/pipeline.rs

pub struct GraphicsPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    device: ash::Device, // cloned handle, kept only to destroy the two objects above
}

impl GraphicsPipeline {
    /// `color_format` should be the swapchain's format (`VulkanSwapchain::format`), matching the
    /// existing pattern of passing swapchain-derived state in rather than the pipeline reaching
    /// into `VulkanSwapchain` itself.
    ///
    /// No `extent` parameter: viewport and scissor are `VK_DYNAMIC_STATE_VIEWPORT`/`_SCISSOR`
    /// (set per-frame via `vkCmdSetViewport`/`vkCmdSetScissor` in `command.rs`, not baked into
    /// the pipeline). Extent can change (window resize); format essentially never does for a
    /// given swapchain. Baking extent in would mean rebuilding the pipeline on every resize on
    /// top of rebuilding the swapchain — dynamic state avoids that for free.
    pub fn new(device: &VulkanDevice, color_format: vk::Format) -> Result<Self>;
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self); // destroys pipeline, then pipeline_layout
}
```

Illustrative shader source (final GLSL is an implementation detail, not part of the spec's contract):

```glsl
// shaders/triangle.vert
#version 450
layout(location = 0) out vec3 fragColor;
vec2 positions[3] = vec2[](vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
vec3 colors[3]    = vec3[](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
```

```glsl
// shaders/triangle.frag
#version 450
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(fragColor, 1.0); }
```

## Vulkan objects

| Object | Created by | Lifetime |
|---|---|---|
| `vk::ShaderModule` (vertex, fragment) | `GraphicsPipeline::new`, from `include_bytes!`'d SPIR-V | Local to `new` — destroyed immediately after `create_graphics_pipelines` returns; a shader module is only needed *during* pipeline creation, never after |
| `vk::PipelineLayout` | `GraphicsPipeline::new` | Owned by `GraphicsPipeline`; empty (no set layouts, no push constant ranges) until Phase 2 |
| `vk::Pipeline` | `GraphicsPipeline::new`, via `VkGraphicsPipelineCreateInfo` with `.push_next(&mut vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&[color_format]))` instead of `.render_pass(...)` | Owned by `GraphicsPipeline` |

**Destroy order**: pipeline before pipeline_layout (standard — a layout can't be destroyed while a pipeline
created from it still exists). Both must be destroyed before `VulkanDevice`'s underlying `VkDevice` is
destroyed — `GraphicsPipeline` must not outlive the `VulkanDevice` it was built from.

**Flagging, not fixing, a related existing issue**: I checked how `VulkanRenderer` in `mod.rs` currently
orders its `instance`/`device`/`swapchain` fields, because this spec's ownership section needs to say
something correct about destroy order. Rust drops struct fields in *declaration* order (verified: top field
drops first), and `VulkanRenderer` declares `instance, device, swapchain` in that order — meaning today,
`instance` (and its debug messenger) is destroyed *before* `device`, which is destroyed *before* `swapchain`
(and the `VkSurfaceKHR` it owns). That's backwards: Vulkan requires every object be destroyed before the
`VkInstance`/`VkDevice` it was created from. This predates this spec and nothing here makes it worse (`
GraphicsPipeline` isn't being wired into `VulkanRenderer` yet — see Non-goals), but it's worth its own small
fix (reverse the field order, or make `VulkanRenderer::Drop` explicit) before `002` adds more objects to that
struct. Not fixing it here since it's outside this spec's stated scope.

## Interactions

- `device.rs`: extension list + feature chaining, as described above.
- `swapchain.rs`: none — `format`/`extent` are already `pub`.
- `error.rs`: none — a malformed/missing `.spv` file can use the existing `InitializationError` variant, no
  new variant needed.
- `Cargo.toml`: no new dependency (`ash::util::read_spv`, used below, ships in `ash` itself).

## Implementation gotcha worth knowing going in

`include_bytes!` gives you `&'static [u8]`, but `VkShaderModuleCreateInfo::code` wants `&[u32]`, and SPIR-V
words are 4-byte little-endian — a raw `bytemuck::cast_slice::<u8, u32>` (tempting, since `bytemuck` is
already a dependency) is only sound if the byte slice happens to be 4-byte aligned, which `include_bytes!`
does **not** guarantee. Use `ash::util::read_spv(&mut Cursor::new(bytes))` instead — it exists specifically
for this, reads the words correctly regardless of the input slice's alignment, and returns `Vec<u32>`. This
is a real "looks fine, is UB" trap, and a good one to have hit once deliberately.

## Test plan

- Unit tests (no live Vulkan required, matching the existing `device.rs`/`swapchain.rs` pattern of testing
  only pure logic):
  - Both checked-in `shaders/*.spv` files parse via `ash::util::read_spv` without error.
  - The parsed word stream starts with the SPIR-V magic number (`0x0723_0203`).
- Manual verification: temporarily construct a `GraphicsPipeline` in `examples/triangle.rs` right after
  swapchain creation, then let it drop before the render loop starts. Run with validation layers on (default
  in debug builds) and confirm **zero** validation errors/warnings in the log output. This confirms the
  pipeline is well-formed and cleanly destroyed *before* `002` adds anything that draws with it — isolating
  "does the pipeline object work" from "does drawing with it work" is the point of scoping this spec the way
  Non-goals does.

## Open questions

- `GraphicsPipeline` holds a cloned `ash::Device` (a flat function-pointer-table copy, not a refcounted
  handle — `ash::Device` isn't `Arc`-backed) purely to call `destroy_pipeline`/`destroy_pipeline_layout` in
  `Drop`. Nothing in the type system stops a caller from dropping the owning `VulkanDevice` first. The
  existing code has the same shape everywhere else (e.g. `VulkanSwapchain` holds its own loader copies) and
  doesn't use `Arc<ash::Device>`, so this spec matches that rather than introducing a new pattern
  unilaterally — but it's worth a deliberate yes/no the first time a *second* Vulkan-object-owning struct
  needs the same handle, rather than three independent implicit "yes"es.

## Outcome

Implemented as specified, no material divergence from the plan:

- **Shaders**: `shaders/triangle.{vert,frag}` compiled with `glslc` to committed `.spv`, loaded via
  `include_bytes!` + `ash::util::read_spv`. (Recompile manually after editing the GLSL — the accepted
  trade-off of precompiled SPIR-V.)
- **`device.rs` prerequisite**: added a `const REQUIRED_DEVICE_EXTENSIONS: &[&CStr]` (swapchain +
  dynamic_rendering) as the single source for both the enabled-extension list and the suitability check
  (removing the duplicated hardcoded check), and chained `PhysicalDeviceDynamicRenderingFeatures` into
  `DeviceCreateInfo` via `push_next`.
- **Pipeline**: dynamic rendering via `PipelineRenderingCreateInfo.color_attachment_formats(&[color_format])`
  pushed into `GraphicsPipelineCreateInfo` (no `.render_pass`); empty vertex input + layout; dynamic
  viewport/scissor; `cull BACK` / `front_face CLOCKWISE`; shader modules destroyed immediately after
  creation; `Drop` destroys pipeline then layout. Signature and `pub(crate)` fields match the spec's
  Public API exactly. Not wired into `VulkanRenderer` (deferred to 002, per Non-goals).
- **Tests**: both `.spv` parse via `read_spv` and start with the SPIR-V magic `0x07230203`.

**Caveat on the manual-verification step.** The test plan assumed "validation layers on (default in debug
builds)", but `instance.rs` currently has an empty `layer_names` — only a debug-utils *messenger* is set up,
the `VK_LAYER_KHRONOS_validation` *layer* is not actually enabled. So the manual run (Apple M5 /
`kosmickrisp` driver) confirms the pipeline **constructs and tears down cleanly with no errors, panics, or
messenger output** — but it is not the full validation sweep the plan implied. Enabling the validation layer
is an `instance.rs` concern affecting the whole renderer; it deserves its own small spec before "zero
validation errors" can be a real acceptance gate. Flagged, not fixed here (out of this spec's scope).

**Open question resolution.** The cloned-`ash::Device` ownership question is left as-is: `GraphicsPipeline`
matches the existing `VulkanSwapchain` pattern (holds a cloned handle, relies on being dropped before the
`VulkanDevice`). No `Arc<ash::Device>` introduced — revisit when 002 adds the next handle-owning struct.
