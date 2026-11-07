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

## Outcome

Pending review and implementation. This is a two-pass contract, not approval to introduce a render graph.

## References

- [Vulkan synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
- [Sampler comparison and LOD configuration](https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerCreateInfo.html)
