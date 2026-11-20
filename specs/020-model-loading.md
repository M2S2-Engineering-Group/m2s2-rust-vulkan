# 020: Model loading (glTF)

**Status**: Draft for review. Not implemented.
**Roadmap item**: Phase 3 "Model loading: glTF (preferred) and/or OBJ" — glTF only for this spec.
**Dependencies**: Spec 007's `Object { model: Mat4, .. }`; spec 010's texture creation; spec 015's
`MeshId`/`create_mesh`. All implemented and merged; this spec adds no new renderer API, only a
standalone data loader that feeds the existing ones.

## Goal

Parse a static (non-animated) glTF file into plain geometry/transform/texture-reference data the
existing `create_mesh`/`create_texture`/`Object` APIs already consume, so externally authored
assets can be rendered without hand-writing vertex arrays in an example. Proven against a real
asset: a legacy mech model from the Mecha-Ziu college project, converted to glTF as a one-off,
out-of-repo step (see Outcome).

## Non-goals

- Animation, skinning, morph targets — entirely ignored, even if present in the source file.
- A scene graph or hierarchical transform system in the library (separate Phase 3 backlog item).
  Each returned mesh carries one fully-resolved world `Mat4`; owning/animating a hierarchy over
  time stays the caller's job.
- A material system (separate Phase 3 backlog item) — only a mesh's single base-color texture
  reference is surfaced. No metallic/roughness factors, normal maps, or multi-texture materials.
- Image decoding inside the library. The loader returns a texture's raw encoded bytes and MIME
  type, undecoded; the caller decodes them (the `png` crate is already a dev-dependency for
  golden-image tests) before calling `create_texture`. Keeps `[dependencies]` free of an image
  codec, consistent with treating asset decoding as a caller/future-companion-crate concern.
- OBJ support (ROADMAP allows "and/or"; deferred to a later spec if wanted).
- External glTF buffer/image URIs (`Source::Uri`) and base64 data URIs — only `.glb`-embedded
  buffer views are supported. `Gltf::open` doesn't auto-resolve external references anyway (see
  Interactions), and every asset this spec ships is a self-contained `.glb`.
- Multiple primitives per mesh, sparse accessors, per-vertex `COLOR_0` (vertices default to white,
  matching existing untextured-geometry convention in `examples/cube.rs`), and vertex counts over
  65535 per primitive (`Index` is `u16` — see Test plan).
- Parsing Mecha-Ziu's legacy `.Model` binary format directly. Out of scope for the *library*
  entirely — but producing the one-off `.glb` this spec's own test/example asset depends on is
  in scope for *this spec's implementation* (see Asset conversion below). The converter script
  itself never enters the library or its normal build; only its output (`MechaZiu.glb` plus the
  embedded texture) is committed, under `examples/assets/mecha_ziu/`.
- Deduplicating textures *within* a single glTF image/material graph beyond straightforward
  index-based sharing (below) — e.g. perceptual/hash-based dedup of visually-identical-but-
  distinct source images is not attempted.

## Public API

```rust
pub struct LoadedModel {
    pub meshes: Vec<LoadedMesh>,
    pub textures: Vec<LoadedImage>,
}

pub struct LoadedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
    pub world_transform: Mat4,
    pub texture: Option<usize>,   // index into LoadedModel::textures
}

pub struct LoadedImage {
    pub bytes: Vec<u8>,     // raw encoded bytes (e.g. PNG), undecoded
    pub mime_type: String,  // e.g. "image/png", as declared by the glTF image
}

pub fn load_gltf(path: &std::path::Path) -> Result<LoadedModel>;
```

`textures` holds each *distinct* glTF image exactly once, keyed internally by the glTF document's
own image index — not once per mesh that references it. This matters concretely for the Mecha-Ziu
asset: all 20 sub-meshes share one base-color texture (the original exporter only ever wrote
`Meshes[0]`'s texture name — see Outcome), so a naive per-mesh `Option<LoadedImage>` would decode
and GPU-upload the same 512×512 PNG 20 times over. Caller loop: decode each `LoadedModel::textures`
entry once to RGBA8 and call `create_texture` once per entry, building a `Vec<TextureId>` parallel
to `textures`; then for each `LoadedMesh`, call `create_mesh(vertices, indices)` and push
`Object { mesh, model: world_transform, texture: texture.map(|i| texture_ids[i]), specular:
SpecularMaterial::default() }`.

A primitive missing a required `POSITION`, `NORMAL`, or `TEXCOORD_0` accessor, or a mesh with more
than one primitive, is rejected with a clear `Err`, matching the external-URI-image rejection
stance above — never defaulted, silently dropped, or taking just the first primitive.

## Vulkan objects

None. This is a pure CPU-side parsing module with no Vulkan handles and no dependency on
`VulkanDevice`/`VulkanRenderer` — it can be unit-tested in CI without a GPU, same as the existing
pure-logic modules (`device.rs::score_device_type`, `mip.rs`'s planning functions).

## Interactions

- Depends on `Vertex`/`Index` (`buffer.rs`) for the returned geometry shape; `Index = u16` means a
  primitive with more than 65535 vertices must be rejected with a clear error, not silently
  truncated or panicking.
- New `[dependencies]` entry: `gltf = { version = "1.4", default-features = false, features =
  ["utils", "names"] }`. Explicitly **not** the crate's default `import` feature — that feature
  unconditionally pulls in the `image` crate for decoding, which is exactly the dependency this
  spec keeps out of `src/`. `Gltf::open(path)` (not `gltf::import`) parses the JSON and, for a
  `.glb`, exposes its embedded binary chunk as `gltf.blob: Option<Vec<u8>>` — no external-URI
  resolution, no image decoding, nothing conditional on the `import` feature.
- Node world transforms: the crate exposes only each `Node`'s local `transform()`; this spec
  hand-rolls the standard recursive composition (walk `document.scenes() -> nodes()`, multiplying
  each ancestor's local matrix) so the loader is correct for any glTF, not just the flat,
  pre-baked-world-matrix shape this spec's own converter happens to produce.
- Primitive attribute/index extraction goes through `mesh::Reader` with a buffer-resolving closure
  backed by `gltf.blob` (no auto-loader): `read_positions()`/`read_normals()`/`read_tex_coords(0)`
  normalize accessor component types to `f32` arrays; `read_indices()...into_u32()` normalizes
  index width, then the loader downcasts to `u16` with the per-primitive vertex-count check above.
- Base-color texture extraction: `primitive.material().pbr_metallic_roughness().base_color_texture()`
  → its image's `Source`. Only `Source::View { view, mime_type }` is supported (see Non-goals);
  the loader slices `view`'s byte range directly out of `gltf.blob`.
- Returned `LoadedMesh`es are consumed entirely by caller code already exercised in
  `examples/cube.rs` (`create_mesh`, `create_texture`, `Object`) — no renderer-side changes.

## Asset conversion

No `.gltf`/`.glb` of the Mecha-Ziu mech exists yet; this spec's manual-verification step depends
on one existing. Producing it is part of this spec's implementation, done once, up front, before
`load_gltf` itself needs to work against a real file:

1. A throwaway script (language/location is implementer's choice — not committed, not part of the
   crate or its build) reads `MechaZiu.Model` using the exact binary layout documented in
   `Mecha-Ziu/Model_Exporter/Model_Exporter/ModelExporter.cpp`'s `WriteModelData_Binary` (texture
   filename, vertex/triangle counts, interleaved `f32` vertex data, `u32` triangle indices, then
   per-sub-mesh name/transform-name/parent-name/counts/world-matrix — animation keyframe data at
   the end is skipped per this spec's non-goals).
2. The script also reads `ziu.tga` (uncompressed 512×512 RGBA Targa, sitting next to the `.Model`
   file) and re-encodes it as PNG.
3. The script emits one self-contained `MechaZiu.glb`: 20 glTF nodes (one per Mecha-Ziu sub-mesh),
   each a mesh/primitive with its own `POSITION`/`NORMAL`/`TEXCOORD_0` accessors and the sub-mesh's
   already-known-good world matrix as the node's `matrix` (no re-derivation of parent-chain math
   needed — see Open questions), all referencing one shared material/base-color image (the
   converted PNG, embedded as a glTF buffer view, not an external URI).
4. `MechaZiu.glb` is committed to `examples/assets/mecha_ziu/`; the script is discarded.

## Test plan

- Unit tests (no GPU, runs in CI): hand-built minimal `.glb` fixtures (committed under
  `testdata/gltf/`) covering: a single node+mesh+texture round-trip; multiple nodes with a parent
  chain (world-transform composition correctness — this is the one actually tricky piece of
  logic); two sibling meshes referencing the *same* image (asserts `textures.len() == 1`, i.e. the
  dedup actually dedupes, not just that both meshes render); a mesh with no base-color texture
  (`texture: None`); a primitive missing `NORMAL` or `TEXCOORD_0` (clean rejection); a mesh with
  more than one primitive (clean rejection); a primitive exceeding 65535 vertices (clean rejection,
  not a panic); and a reference to an external URI image (clean rejection per Non-goals, not a
  silent skip).
- Manual verification: a new `examples/model.rs` loads the converted Mecha-Ziu asset (checked into
  `examples/assets/mecha_ziu/`), decodes its texture via the existing `png` dev-dependency, and
  renders all 20 resulting objects under the existing free-fly camera — confirms the pelvis/torso/
  gun-arm/head/leg hierarchy reassembles correctly on screen from flattened, independent world
  matrices, not just that the loader returns the right *count* of meshes.

## Open questions

- Exact converter output shape (one flat root-level node per sub-mesh with its pre-baked world
  matrix, vs. a real parent/child node chain) — either is spec-compliant since the loader composes
  transforms recursively either way; pick whichever is less code in the throwaway script.
- Where `examples/assets/` lives relative to the crate root and whether `cargo build --examples`
  needs any manifest change to make sure the asset is found relative to the binary rather than the
  caller's CWD — resolve during implementation.

## Outcome

(Filled in after implementation.)
