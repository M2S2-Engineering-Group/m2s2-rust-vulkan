# 010: Textures

**Status**: Drafted; not yet implemented.
**Roadmap item**: [Phase 2](../ROADMAP.md#phase-2-standard-rendering-features), "Textures".
**Prerequisite**: specs 006–009 implemented. Builds directly on the sampled-R8 image path and shared
one-time upload introduced for text (spec 008): `src/renderer/image.rs` and `src/renderer/upload.rs`.

## Goal

Let a caller upload RGBA8 textures and draw each scene object sampled from its own texture, combined
with the existing per-vertex color. One camera and one mesh as before; each drawn copy may reference a
different texture.

## Non-goals

No image-file decoding (PNG/JPEG) in this crate — the caller supplies already-decoded RGBA8 bytes.
Decoding, and other "game engine but not Vulkan" helpers, belong to a future separate utilities crate
with its own spec, not here. Also out of scope: mipmaps (deferred to a follow-up spec), multiple
meshes, materials, lighting, anisotropic filtering, and mid-run texture freeing. This spec introduces
a minimal per-type texture handle (`TextureId`), not the general resource-handle system slated for
Phase 3 — the latter (handles across buffers/images/pipelines, generational validity) stays deferred.

## Public API

Textures are renderer-owned GPU resources referenced by an opaque, `Copy` handle. The caller never
holds the GPU resource directly, so there is no cross-object lifetime to manage and nothing to free by
hand — the renderer's `Drop` reclaims every texture. Re-export the handle and the draw type from the
crate root.

```rust
/// Opaque handle to a renderer-owned texture, returned by `create_texture`. `Copy` and cheap to
/// pass around; valid for the lifetime of the renderer that produced it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TextureId(/* private index */);

/// One drawn copy of the uploaded mesh: a model transform and an optional texture.
/// `None` samples a renderer-owned 1x1 opaque-white texture, so an untextured object
/// reproduces the pre-010 look (`white * vertexColor == vertexColor`).
pub struct Object {
    pub model: Mat4,
    pub texture: Option<TextureId>,
}

impl VulkanRenderer {
    /// Upload an RGBA8 image (`rgba8.len() == width * height * 4`, both non-zero) into a new
    /// device-local sampled texture owned by the renderer, returning its handle. Validation error
    /// otherwise, before any GPU work.
    pub fn create_texture(&mut self, rgba8: &[u8], width: u32, height: u32) -> Result<TextureId>;

    pub fn render_frame(&mut self, model: Mat4, view: Mat4, proj: Mat4) -> Result<()>;
    pub fn render_objects(&mut self, objects: &[Object], view: Mat4, proj: Mat4) -> Result<()>;
    pub fn render_objects_with_text(
        &mut self, objects: &[Object], view: Mat4, proj: Mat4, text: &[TextDraw<'_>],
    ) -> Result<()>;
}
```

`render_frame(model, view, proj)` forwards as `[Object { model, texture: None }]`. `render_objects`
forwards to `render_objects_with_text` with empty text. One draw concept (`Object`) across the whole
family — no parallel textured/untextured method set. An empty slice still clears and presents.

There is no `destroy_texture` in this spec: the Phase 2 milestone is a static textured scene, nothing
churns textures, and every texture is freed when the renderer drops. Mid-run freeing is a later
additive method (backed by a generational slot map), explicitly out of scope here.

**Breaking change** (intentional, pre-1.0, only in-repo consumers): `render_objects` and
`render_objects_with_text` take `&[Object]` instead of `&[Mat4]`. `examples/cube.rs` updates to pass
`Object`s; `examples/triangle.rs` keeps using `render_frame` unchanged.

## Vulkan objects

### Vertex layout
`Vertex` gains `uv: Vec2` after `color` (so existing field order is preserved): `position` (offset 0),
`color` (offset 12), `uv` (offset 24), size 32. `attribute_descriptions` returns three entries, adding
`location = 2`, `R32G32_SFLOAT` at `offset_of!(Vertex, uv)`.

### Texture format and color space
Textures use `R8G8B8A8_SRGB`. The swapchain is already sRGB (`swapchain.rs` prefers it), so hardware
linearizes texels on sample and re-encodes on store — albedo composites in linear space. Document this
choice; it is the correct default for color images and the reason the format is not `UNORM`.

### Descriptor sets
The camera UBO stays at set 0. Add a second descriptor set, **set 1**, binding 0: a
`COMBINED_IMAGE_SAMPLER` visible to the fragment stage. The renderer owns the set-1
`DescriptorSetLayout` (so it exists before both pipeline creation and texture creation) and passes it
as `set_layouts[1]` to the scene pipeline layout. Each stored texture owns a small descriptor pool
(`max_sets = 1`, one combined image sampler) and the set allocated from it against that shared layout,
written to point at that texture's view + sampler.

The text pipeline is unaffected: it has its own layout with the font atlas at its own set 0.

### Upload path
Generalize the existing R8 path in `image.rs` into a shared sampled-image constructor parameterized by
format and bytes-per-pixel, with `new_r8` (font) and `new_rgba8_srgb` (textures) as thin callers — do
not fork a second uploader. Keep the `UNDEFINED -> TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL`
transitions and the transfer-write -> fragment-read visibility already used for the font, via
`upload::submit_once`. Sampler: linear min/mag filter, `CLAMP_TO_EDGE` (nearest stays the font's
choice). Single mip level.

### Default white texture
The renderer creates one 1x1 opaque-white (`[255, 255, 255, 255]`) texture at construction, held
separately from the caller-facing texture store (it has no `TextureId`), and used whenever
`Object.texture` is `None`. Built through the same `new_rgba8_srgb` path.

## Recording

Camera descriptor set (set 0), pipeline, vertex and index buffers are bound once per frame as today.
Before recording, resolve each `Object` to a `(model, texture_descriptor_set)` pair — `None` resolves
to the white texture's set; `Some(id)` resolves to the stored texture's set at that index. Extend the
shared `emit_object_draws` command path (spec 007) with a
`BindTexture(vk::DescriptorSet)` command so one path still serves both production and the CPU recording
tests: per object emit `BindTexture(set)` (set 1), `PushModel(bytes)`, `DrawIndexed(count)`. No change
to text recording, which still follows scene geometry in the same dynamic-rendering scope.

## Shaders

`shaders/triangle.vert`: add `layout(location = 2) in vec2 inUV;` and pass it through as
`layout(location = 1) out vec2 fragUV;` (color stays `location = 0`). `shaders/triangle.frag`: add
`layout(set = 1, binding = 0) uniform sampler2D tex;` and compute
`outColor = texture(tex, fragUV) * vec4(fragColor, 1.0);`. Recompile and commit the SPIR-V with the
existing `glslc` workflow and validate with `spirv-val`.

## Ownership and teardown

The renderer owns all textures: a `Vec` of stored textures (indexed by `TextureId`) plus the separate
white default. A stored texture is not `Drop` (freeing its image/allocation/pool needs `&Device` +
`&mut Allocator`), so the renderer frees them explicitly in its own `Drop`, matching how `Buffer`,
`depth`, and `uniforms` are already handled — the caller does nothing and cannot leak or dangle. In
`Drop`, the renderer destroys every stored texture and the white texture, plus the set-1 layout,
within the existing ordered teardown: all texture images/allocations/pools freed before the allocator,
and the allocator before the device. `create_texture` failure releases any partially created resources
(reuse the image constructor's existing rollback) and adds nothing to the store.

## Interactions

- `buffer.rs`: add `uv` to `Vertex` and a third attribute description; update the doc comment and the
  packing test.
- `image.rs`: parameterize the sampled-image constructor by format/bpp; add `new_rgba8_srgb`; extend
  `validate_pixels` to bytes-per-pixel.
- `pipeline.rs`: scene pipeline layout gains `set_layouts[1]` (the texture layout). Text pipeline
  unchanged.
- `command.rs`: `BindTexture` command in `emit_object_draws`; record bind-set-1 per object.
- `mod.rs`: own the set-1 layout, the white texture, and the `Vec` texture store; `TextureId`,
  `Object`, `create_texture`; the `Option<TextureId>`-to-descriptor-set resolution; convert the draw
  methods to `&[Object]`; teardown additions.
- `lib.rs`: re-export `TextureId` and `Object`.
- `shaders/triangle.{vert,frag}` + committed `.spv`.
- `examples/cube.rs`: pass `Object`s; upload at least two distinct small procedurally-generated RGBA8
  textures (no asset files) and show per-object textures plus one `None` (white) object.
- `examples/triangle.rs`: unchanged (`render_frame`).

## Test plan

CPU-only (CI has no driver):
- `Vertex` size 32 and offsets (position 0, color 12, uv 24); three attribute descriptions with
  `location = 2`, `R32G32_SFLOAT`, offset 24.
- `validate_pixels` for 4 bpp: accepts `w*h*4 == len`, rejects mismatch and zero extent.
- `Option<TextureId>`-to-set resolution: `None` resolves to the white set; `Some(id)` resolves to the
  stored set at that index (test the pure resolution helper against a small table of sentinel
  descriptor-set values).
- `emit_object_draws`: empty slice emits nothing; one object emits `BindTexture, PushModel, DrawIndexed`
  in that order; two objects with distinct sets/models emit two correct, correctly ordered triples.
- Scene vertex/fragment SPIR-V parse (magic word) after recompilation.

Manual GPU checks with synchronization validation enabled:
- Two objects each sampling a different uploaded texture, combined with vertex color, under one camera.
- A `None` object renders exactly as pre-010 (vertex colors), proving the white default.
- Depth occlusion unchanged; repeated resize, minimize/restore, and shutdown produce no validation
  errors; renderer teardown with textures in the store leaks nothing (gpu-allocator clean on drop).

Run all four repository quality gates. Check off "Textures" in the roadmap only after implementation
and verification; mipmaps and instancing remain deferred.

## Open questions

None for this scope. Mipmaps, a utilities/decoding crate, and camera controllers are tracked as
separate future efforts.

## Outcome

_To be recorded after implementation and the GPU regression check._
