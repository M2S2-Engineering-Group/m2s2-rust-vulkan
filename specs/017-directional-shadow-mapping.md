# 017: Directional shadow mapping

**Status**: Draft for review; pass-recording interfaces are provisional. Not implemented.
**Roadmap item**: Phase 4 shadow mapping, pulled forward as a bounded first multi-pass example.
**Dependencies**: 012 ownership/error handling; 011/013 directional shading; 015 mesh resources.
Use 016's instance batches if available. Do not build a second mesh/object resolver for the shadow pass.
014 mipmaps are independent. This spec does not require a full render graph.

## Goal

Render a directional light's depth map, then sample it during the main scene pass so geometry casts
hard shadows onto other geometry. Establish explicit pass/resource dependencies with dynamic rendering.

## Non-goals

Cascades, camera-frustum fitting, point/spot shadows, PCF/soft shadows, transparent/alpha-tested casters,
shadow atlases, a render graph, multithreaded recording, or automatic scene bounds. Current geometry is
opaque; texture alpha is not a shadow cutoff. A later spec must address cutout materials explicitly.

## Public API: proposed contract

```rust
#[derive(Copy, Clone, Debug)]
pub struct DirectionalShadow {
    pub light_view_proj: Mat4,
    pub resolution: u32,       // square depth map, proposed example value 1024
    pub constant_bias: f32,    // raster depth-bias constant factor
    pub slope_bias: f32,       // raster depth-bias slope factor
}

impl VulkanRenderer {
    pub fn set_directional_shadow(&mut self, shadow: Option<DirectionalShadow>) -> Result<()>;
}
```

Default None; existing scenes have no shadow pass. Caller supplies a finite light view-projection
matrix using the renderer's Vulkan clip conventions (depth [0,1] and exactly one Y adjustment), aimed
consistently with the configured sun. The library does not infer a light camera from a direction alone.
No general camera controller or automatic shadow fitting enters the library.

Validate nonzero resolution, device/image-format limits, finite matrix/bias values, and required sampled
depth capabilities. Reject invalid configuration without changing the prior setting. Bias factors may
be signed finite values; default example uses small positive factors, recorded after visual testing.
Setting configuration stores a validated pending target. Applying a new resolution allocates complete
replacement resources before retiring the old resources after confirmed completion. Report allocation
failure from the next render if resources are created lazily; keep prior ownership valid for teardown.

Exact setter timing and private pass method signatures must be finalized after reviewing 012/015/016.
This draft fixes ownership and ordering requirements without inventing a general pass API now.

## Resources and ownership

Use one shadow depth image/allocation/view and descriptor set per frame slot, plus per-frame light
view-projection uniform buffers. Proposed format priority: D32_SFLOAT, then D16_UNORM, requiring both
DEPTH_STENCIL_ATTACHMENT and SAMPLED_IMAGE support with optimal tiling. Avoid stencil formats and
unenabled separate-depth/stencil-layout features. Each image is single-sample, one level, with
DEPTH_STENCIL_ATTACHMENT | SAMPLED usage. Resolution is independent of swapchain extent.

Set 3 is the shadow set: binding 0 combined depth sampler (FRAGMENT), binding 1 light-view-projection
UBO (VERTEX | FRAGMENT). Light matrix payload is one row-major mat4 (64 bytes). Use a nearest comparison
sampler, compare LESS_OR_EQUAL, normalized coordinates, no anisotropy, LOD 0. Hard single-tap comparison
avoids requiring depth linear-filter support. Validate the actual selected format/usage configuration.

Use shadow-enabled/disabled scene shader variants built from one shared source path. Disabled scenes
retain the non-shadow pipeline/layout and do not need a dummy sampled depth image or sampled-depth
support. Enabled scenes use set 3 and a depth-only shadow pipeline. Do not duplicate the full lighting
implementation across variants. A disabled feature must work on a device lacking shadow capabilities.

All resources remain renderer-owned and private. Follow 012 for constructor rollback, exclusive mapped
updates, confirmed-completion destruction, and error retention. Disabling skips future shadow passes;
retire any previously used resources only after confirmed completion. Swapchain resize updates main
pass state but does not regenerate the independently sized shadow maps.

## Explicit two-pass execution

Record both passes in one primary command buffer on the graphics queue. Start with private
record_shadow_pass / record_scene_pass functions sharing resolved draw data; their exact signatures
are provisional. No VkRenderPass/VkFramebuffer objects are needed for dynamic rendering.

For the selected frame slot, wait for its fence before mapped updates or shadow-image reuse:
1. Transition its depth image from UNDEFINED on first use, or DEPTH_STENCIL_READ_ONLY_OPTIMAL on reuse,
   to DEPTH_STENCIL_ATTACHMENT_OPTIMAL. On reuse, order prior fragment sampling before attachment
   read/write at EARLY | LATE_FRAGMENT_TESTS. Track initialized/layout state explicitly.
2. Begin depth-only dynamic rendering at shadow resolution; clear depth to 1, STORE for sampling.
   Set viewport/scissor, enable depth test/write LESS and dynamic depth bias. Draw all scene casters
   using their model transforms and light_view_proj. Use back-face culling initially. Do not sample
   the shadow image during this pass or accidentally use the main camera projection.
3. End rendering. Transition depth attachment -> DEPTH_STENCIL_READ_ONLY_OPTIMAL, source
   EARLY | LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_WRITE, destination FRAGMENT_SHADER /
   SHADER_READ. A completed recording operation alone is not a GPU dependency.
4. Begin the existing main scene rendering scope. Restore main viewport/scissor and scene pipeline
   state explicitly; bind the selected frame's shadow set. Draw receivers, then the existing text.
5. Present through the existing acquire/submit/present path. One frame fence covers both passes.

Pass failure must follow 012's frame-error state; no retry with partially recorded state or an
unsignaled fence. The image count and frame-slot count must not be conflated.

## Sampling and lighting

Project world position by light_view_proj, divide by w only when w is positive/nondegenerate, and map
x/y from [-1,1] to normalized texture coordinates. Light projection already follows Vulkan conventions;
do not flip Y a second time. Positions outside x/y bounds or z outside [0,1] are explicitly fully lit.
Sample the comparison texture with the receiver depth; apply raster bias once, not a second ad hoc
shader bias. Document depth convention and comparison direction alongside the shader.

Only directional diffuse and specular contributions are multiplied by shadow visibility. Ambient
remains unshadowed. Later point lights (018) are unshadowed by this directional map. Specular-off
materials and text preserve their earlier behavior. Shadow draws are excluded from scene diagnostics;
existing scene draw/index/instance counters describe the main scene only.

## Test plan

CPU: configuration validation, capability fallback/rejection, resource selection per frame slot,
light-matrix layout, first-use/reuse barrier scopes and ranges, pass order, viewport restoration,
disabled variant with missing shadow support, unchanged prior configuration on rejection, and resource
replacement/failure retention via 012's actual backend. Reuse production draw plans for both passes.

GPU: a cube casting onto a separate plane; move camera while light remains stationary; move the light
and matching projection; verify disabled output against the previous baseline; inspect real shadow depth
contents; test out-of-frustum receivers, unlit ambient, bias tradeoffs/acne/peter-panning, resolution
changes, instanced casters when 016 is present, text, resize/minimize, and teardown. Require no validation
or synchronization errors with two frames in flight. Record device, depth format, settings, comparison
tolerance, and visual artifacts in Outcome. A lit screenshot alone does not prove the shadow pass works.

Run all quality gates/doctests and compile/validate every scene variant and depth shader. CPU-only CI
must not load a Vulkan driver.

## Open questions for review

- Finalize private pass-recording boundaries after the mesh/instance plan is accepted.
- Accept caller-supplied light projection, hard nearest shadows, and per-frame depth images initially?
- Agree on configuration-application timing, bias defaults, and GPU reference images before coding.

## Resolved decisions (2026-11-19)

Grounded against the actual current code (`mod.rs`, `command.rs`, `pipeline.rs`, `depth.rs`,
`light.rs`, `uniform.rs`, `per_frame_uniform.rs`, `shaders/triangle.{vert,frag}`) rather than the
spec's abstract description, following this project's established per-spec practice.

**Caller-supplied projection, hard nearest shadows, per-frame depth images: accepted as proposed.**
No camera-fitting, no PCF, one depth image per frame-in-flight (`FRAMES_IN_FLIGHT = 2`,
`command.rs:20`).

### Shared draw-emission path (no second resolver)

`command::emit_object_draws` (`command.rs:565`) already walks `objects` once and emits
pass-agnostic `ObjectCommand`s (`BindInstanceBuffer`/`BindMesh`/`BindTexture`/`DrawIndexed`) through
a caller-supplied closure — the main scene pass's real Vulkan recording (`command.rs:456-494`), the
stats re-walk (`mod.rs:511-532`), and this crate's own tests already reuse it with different
closures. The shadow pass calls this **same function again**, unchanged, with its own closure: binds
the identical mesh/instance buffers, **no-ops on `BindTexture`** (the shadow pipeline has no texture
set to bind into), and draws with the exact `instance_count`/`first_instance` the main pass uses —
so instanced casters (spec 016) are covered for free. This satisfies "do not build a second
mesh/object resolver" and "reuse production draw plans for both passes" with zero changes to
`emit_object_draws` itself.

### Shader variants: one source, two compiled artifacts

`CLAUDE.md`'s shader recompile note (`glslc shaders/triangle.vert -o shaders/triangle.vert.spv`) has
no macro-flag precedent. Resolved: `triangle.vert`/`triangle.frag` gain `#ifdef SHADOWS_ENABLED`
blocks (light-space position output in the vertex shader; shadow-map sampling + comparison in the
fragment shader) and are compiled **twice** — once unchanged (`triangle.vert.spv`/`triangle.frag.spv`,
the existing disabled-shadow artifacts, byte-identical to today since the macro is simply undefined)
and once with `-DSHADOWS_ENABLED` into new `triangle_shadow.vert.spv`/`triangle_shadow.frag.spv`. One
shared source path, two `GraphicsPipeline`s, no duplicated lighting math. `CLAUDE.md`'s recompile
instructions gain the second `glslc -DSHADOWS_ENABLED ...` invocation.

### Pipeline-layer generalization (foundation, zero behavior change)

`GraphicsPipeline::create` (`pipeline.rs:156`) currently requires exactly one `color_format:
vk::Format` and an unconditional `fragment_shader: &'a [u8]` in `PipelineDescription`. The shadow
pass's own pipeline is depth-only: **no color attachment, no fragment shader** (a legal, common
Vulkan shadow-map pattern). Before any shadow-specific code lands, a dedicated zero-behavior-change
task widens `PipelineDescription.fragment_shader` to `Option<&'a [u8]>` (skip the fragment stage
entirely when `None`) and `create`'s `color_format: vk::Format` to `color_formats: &[vk::Format]`
(empty slice = depth-only `PipelineRenderingCreateInfo`, no `.color_attachment_formats(...)` call).
The two existing call sites (`GraphicsPipeline::new` in `pipeline.rs:124`, and `text.rs:166`'s
`GraphicsPipeline::create` call) are updated to pass `Some(FRAGMENT_SHADER)`/`&[color_format]` with
**no observable behavior change** — this mirrors spec 016's Task 2 (the `PipelineDescription.binding`
→ `bindings` plural change), done specifically so a later task's real feature work isn't also the
first commit to touch this shared shape.

### Descriptor set 3 (the shadow set) and the shadow pipeline's own layout

Set 3: binding 0 combined depth-comparison sampler (FRAGMENT only — only the main pass samples it),
binding 1 light-view-projection UBO, 64 bytes row-major (VERTEX | FRAGMENT — the main pass's vertex
shader needs it to compute each vertex's light-space position, and the shadow pass's OWN vertex
shader needs it to transform vertices into light clip space). This is genuinely a two-binding set of
different descriptor types, so it does **not** fit `per_frame_uniform.rs`'s `PerFrameUniform`
(single-binding, `UNIFORM_BUFFER`-only, confirmed by reading `per_frame_uniform.rs:35-39`) — a new
bespoke module (`shadow.rs`) builds its own layout/pool, following the same per-frame-array shape
`PerFrameUniform`/`Light` already establish, not inheriting from either.

**The shadow pass's pipeline layout is `[set 3 only]`** — one set, not four. The depth-only shadow
pass needs nothing from camera (set 0), texture (set 1), or light/point-lights (set 2): the model
transform arrives via the existing per-instance vertex attributes (`InstanceData.model_row0..3`,
already written into the instance buffer before either pass records, per `command.rs:211-221`), and
the only "camera" the shadow pass has is the light's own view-projection, read from set 3 binding 1.

**The main scene pipeline gains set 3 only when shadows are enabled** — two full pipeline/shader
variants, selected exactly like the existing resize path already conditionally rebuilds the pipeline
when `self.pipeline_formats` changes (`mod.rs:450-467`): generalize that rebuild trigger to also fire
on an enabled/disabled shadow transition. This is required, not just an optimization — continuing to
use the shadow-enabled fragment shader (which statically declares set 3) while shadows are disabled
would mean sampling an unbound descriptor set, which is invalid regardless of whether the shader
path is dynamically reached.

### Configuration timing: resolution/enable-state reallocates; light_view_proj/bias update per-frame

`DirectionalShadow` bundles `resolution` with `light_view_proj`/`constant_bias`/`slope_bias`, but
these change at very different rates: a moving/rotating light updates `light_view_proj` every frame,
while `resolution` changes rarely (if ever, post-setup). Resolved: `set_directional_shadow` validates
the whole struct up front (as the spec requires — nonzero resolution, finite matrix/bias), but only
reallocates GPU resources (depth images/views/pipeline/descriptor sets) when `resolution` changes (or
shadows transition enabled↔disabled) — never on every frame's call for a moving light. This mirrors
the existing split between `pending_resize` (deferred, reallocates) and
`directional_light`/`point_lights` (validated immediately, uploaded into the per-frame UBO on every
render like `Light::update` already does — `light.rs:218-226`). Concretely: `VulkanRenderer` gains a
`pending_shadow_config: Option<Option<ShadowResourceConfig>>` (outer `Option` = "a reallocation is
pending"; inner = enable-at-this-resolution vs. disable) consumed at the same point in
`render_objects_with_text` the existing `pending_resize` is (`mod.rs:418-469`), plus a
`current_shadow_values: Option<(Mat4, f32, f32)>` (light_view_proj, constant_bias, slope_bias)
uploaded into set 3's per-frame UBO every render exactly like `directional_light` is today — no
reallocation on that path. Avoiding an accidental "reallocate every frame" trap this split prevents
is worth documenting explicitly since it's not literally spelled out by the spec's setter signature.

### First-use vs. reuse layout tracking

Each of the `FRAMES_IN_FLIGHT` shadow depth images needs an explicit per-slot flag (`[bool;
FRAMES_IN_FLIGHT]` on the new `shadow.rs` resource struct) recording whether that slot has ever been
rendered into: `false` (fresh allocation, image is `UNDEFINED`) until its first successful shadow-pass
recording, then `true` thereafter (meaning the main pass left it in
`DEPTH_STENCIL_READ_ONLY_OPTIMAL`, which the next shadow-pass use must transition back from). A
resolution change (full reallocation) resets all slots to `false`. This is the explicit state the
spec's barrier-scope section requires ("Track initialized/layout state explicitly") and mirrors the
same first-use/steady-state distinction `depth.rs`'s existing barrier already encodes implicitly via
`UNDEFINED` (there, every frame discards and reclears, so there's no reuse case to track — the
shadow map's "read-only after sampling" state is the genuinely new wrinkle).

### `HeadlessRenderer` mirroring

`HeadlessRenderer` (`headless.rs`) has its own independent recording function
(`render_frame_with_text`, not a call into `command::FrameContext`) but already shares
`command::emit_object_draws` for the per-object walk. It needs the same shadow fields/resources as
`VulkanRenderer` and its own call into a shared `record_shadow_pass` helper (living in `command.rs`,
parameterized over the raw `ash::Device`/`vk::CommandBuffer` so both `FrameContext::record` and
`HeadlessRenderer`'s recording function call the identical barrier/begin-rendering/draw-emission
sequence) — this is the concrete form of the spec's "private record_shadow_pass... sharing resolved
draw data."

## Outcome

Pending implementation. This is a two-pass contract, not approval to introduce a render graph.

## References

- [Vulkan synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
- [Sampler comparison and LOD configuration](https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerCreateInfo.html)
