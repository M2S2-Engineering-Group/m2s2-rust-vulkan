# 014: Texture mipmaps

**Status**: Draft for review; not implemented.
**Roadmap item**: [Phase 2 textures](../ROADMAP.md#phase-2-standard-rendering-features), mipmaps deferred
by spec 010.
**Prerequisites**: Spec 010 texture upload and the ownership/upload-failure contracts agreed in spec
012. Independent of spec 013; either rendering feature can be implemented first after cleanup.

## Goal

Generate full mip chains for opted-in RGBA8 sRGB textures and sample between levels during
minification, reducing distant texture aliasing. Reuse the existing sampled-image and upload paths.
Keep text atlas sampling and existing texture calls predictable.

## Non-goals

Anisotropic filtering, compressed textures, array/cube/3D images, CPU/compute mip generation, authored
mip chains, streaming, general sampler configuration, texture unloading, alpha-coverage preservation,
and normal-map filtering. No new device feature or crate dependency.

## Public API

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mipmaps {
    Disabled,
    Generate,
}

#[derive(Copy, Clone, Debug)]
pub struct TextureOptions {
    pub mipmaps: Mipmaps,
}

impl Default for TextureOptions {
    // Disabled, preserving the existing single-level texture behavior.
    fn default() -> Self;
}

impl VulkanRenderer {
    pub fn create_texture(
        &mut self, rgba8: &[u8], width: u32, height: u32,
    ) -> Result<TextureId>;

    pub fn create_texture_with_options(
        &mut self, rgba8: &[u8], width: u32, height: u32, options: TextureOptions,
    ) -> Result<TextureId>;
}
```

Re-export Mipmaps and TextureOptions. The existing create_texture delegates with explicit default
options; both methods share validation/allocation/upload logic. Object, TextureId, descriptor bindings,
and draw methods do not change. Texture data remains the tightly packed RGBA8 base level only.

Generate is an explicit request. If a multi-level linear-blit path is unsupported, return a descriptive
initialization error; do not silently create one level, change format/filtering, or select a different
GPU. The caller can explicitly retry Disabled. A generated 1x1 texture has one level and needs no blit.
No partial TextureId or descriptor-store entry is published on failure.

## Mip dimensions and validation

For nonzero dimensions, levels = floor(log2(max(width,height))) + 1. Compute this with integer bit
operations. At level i, each dimension is max(1, base_dimension >> i), implemented without invalid
shifts. Examples: 1x1 -> one level; 8x4 -> 8x4,4x2,2x1,1x1; 7x3 -> 7x3,3x1,1x1; 1x8 -> 1x8,1x4,1x2,1x1.
Disabled always allocates one level.

Validate nonzero dimensions and exact base byte count with checked arithmetic before allocation.
Reject arithmetic overflow, sizes above selected-device/image-format limits, and chains exceeding
supported mip counts. Use the VkImage memory requirements for allocation; do not calculate device
allocation size by summing raw texel bytes.

## Capability checks and explicit failure policy

Query the selected device for optimal-tiled R8G8B8A8_SRGB support. For more than one generated level,
require BLIT_SRC, BLIT_DST, and SAMPLED_IMAGE_FILTER_LINEAR format features, plus sampled-image support.
Check image-format support for the actual usage/tiling/extent/mip configuration and device limits.
The selected graphics queue is used for blits. Do not enable sampler anisotropy or other unused features.

Missing capabilities must fail before queue submission. Keep capability selection in a pure helper
fed by queried properties so positive/negative combinations can be tested without a Vulkan driver.
The one-level Disabled/font paths must not acquire an unnecessary multi-level blit requirement.

## Vulkan objects and upload sequence

Extend the current shared SampledImage construction with a named internal mip policy. The R8 font
atlas explicitly requests one level, nearest min/mag filtering, and nearest mip mode. The white default
texture also remains one level. Do not create a font-specific duplicate uploader or another allocator.

A generated scene texture uses:
- Image: R8G8B8A8_SRGB, optimal tiling, single sample, complete level count, SAMPLED | TRANSFER_DST and
  TRANSFER_SRC when multiple levels will be generated.
- View: baseMipLevel 0, levelCount equal to all allocated levels, COLOR aspect, one array layer.
- Sampler: LINEAR min/mag, LINEAR mipmap mode, minLod 0, maxLod level_count - 1, bias 0, anisotropy off.
  A one-level chain has maxLod 0; Disabled preserves its existing single-level filtering settings.

Record the entire upload in the existing one-time command submission:
1. Transition all allocated levels from UNDEFINED to TRANSFER_DST_OPTIMAL with transfer-write access.
2. Copy the caller's base pixels from staging into mip 0 only.
3. For each destination level i from 1:
   - Transition source level i-1 from TRANSFER_DST_OPTIMAL to TRANSFER_SRC_OPTIMAL, ordering its prior
     transfer write before the upcoming transfer read.
   - Linear-blit from level i-1 to level i using their independently clamped dimensions.
   - Transition the consumed source level from TRANSFER_SRC_OPTIMAL to SHADER_READ_ONLY_OPTIMAL,
     ordering transfer-read use before fragment shader reads.
4. Transition the final level from TRANSFER_DST_OPTIMAL to SHADER_READ_ONLY_OPTIMAL, ordering its
   transfer write before shader reads. This step also handles a one-level upload with no blits.

Use explicit source/destination subresource levels and disjoint ranges for each blit. sRGB conversion
and filtering follow Vulkan's blit rules; do not manually average encoded sRGB bytes or switch to an
UNORM image as a hidden workaround. Ordinary shader texture() calls select LOD from derivatives; no
production shader change is required.

One-level view/sampler ranges must never expose uninitialized levels. No texture is published until
all levels are initialized and upload completion is confirmed.

## Ownership, failure, and teardown

The image/allocation still belongs to its StoredTexture; the view covers its mip subresources and the
sampler selects among them. Descriptor set 1 remains one combined image sampler. Mip levels do not
create separate image allocations or per-level public handles.

Apply spec 012's consuming cleanup and pending-upload ownership. Staging, destination image/allocation,
command pool, and fence remain alive until completion is established. A wait error after submission
must not trigger immediate rollback of in-use resources. Constructor failure before submission releases
completed objects once; descriptor creation/store insertion happens only after a complete upload.

Normal teardown remains descriptor resources, sampler/view, image, allocation, then allocator/device
parents after completion. Resize does not regenerate textures or their mip chains.

## Interactions and demonstration

- `image.rs`: mip policy, complete image/view/sampler ranges, capability validation, per-level blits
  and barriers. Use shared mip-extent helpers for allocation and command recording.
- `upload.rs`: reuse the completion/retention contract from 012; no second submission implementation.
- `texture.rs`/`mod.rs`/`lib.rs`: options forwarding and publication after success.
- `text.rs`: explicitly preserve the one-level nearest font path.
- Cube example: upload a sufficiently detailed procedural checker (proposed 256x256 with repeated
  4x4-texel checks), compare identical textures with Disabled and Generate, and move the camera from
  near to far. The current 2x2 checker is not an adequate minification demonstration.
- Keep diagnostics and lighting unchanged. Label comparison objects or explain which is mipmapped;
  do not add a general interactive sampler UI.

## Test plan

CPU-only:
- Level counts/extents for 1x1, powers of two, odd dimensions, tall/wide images, and zero rejection.
- Checked base-byte arithmetic, mismatched pixel bytes, oversized dimensions and mip-limit rejection.
- Default/Disabled path versus Generate; generated 1x1 succeeds without blit capabilities.
- Capability helper rejects each missing required feature and accepts the full supported combination.
- Test actual command-planning/recording helpers: one-level uploads emit no blits; N levels emit N-1
  blits; ranges/dimensions and transfer-read/write scopes match every level; all final layouts are
  shader-readable. Do not test an independent simulated mip algorithm.
- View/sampler ranges cover exactly initialized levels. Font and default-white policies stay one level.
- Inject upload/descriptor failure: no published handle/store entry, one-time rollback before submit,
  and resource retention after submit when completion has not been established, as specified by 012.

GPU opt-in: inspect actual mip contents/levels in a frame capture or readback; compare distant
high-frequency texture stability with generation disabled/enabled. Check near-field appearance,
odd/narrow texture dimensions, 1x1, text clarity, dynamic texture creation between frames, resize and
normal teardown under synchronization validation. Include an sRGB-aware expected-value test with a
recorded tolerance; GPU filtering is not required to match arbitrary CPU rounding byte-for-byte.
Capability failure must be tested with injected properties if the available GPU supports everything.

Run all repository quality gates and doctests. Validate any test shaders used for mip readback. Ordinary
CPU CI must not require a Vulkan driver. Record tested hardware, capability path, and visual/readback
results in Outcome; a passing CPU planner test does not prove the GPU mip chain is correct.

## Open questions for review

- Accept Disabled as the compatibility default and explicit Generate opt-in?
- Accept a descriptive unsupported error with caller-directed retry, rather than a CPU fallback?
- Choose the GPU readback fixture and filtering tolerance before implementation acceptance.

## Outcome

Pending review and implementation. No upload or rendering behavior is changed by this draft.

## References

- [Vulkan blit requirements, filtering, and layouts](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBlitImage.html)
- [Sampler mipmap mode and LOD bounds](https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerCreateInfo.html)
- Specs 008, 010, and draft 012 for font/texture uploads and pending-resource ownership.
