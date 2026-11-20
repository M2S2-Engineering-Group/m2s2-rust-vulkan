# 020: Model loading (glTF)

**Status**: Implemented and verified (unit tests + real-asset regression test pass; windowed
example launches and runs error-free; see Outcome).
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

Implemented as drafted, with one shape change made during the builder-check review (before any
code was written): `load_gltf` returns `LoadedModel { meshes: Vec<LoadedMesh>, textures:
Vec<LoadedImage> }` rather than a bare `Vec<LoadedMesh>` with an inline `Option<LoadedImage>` per
mesh, so a texture shared by several meshes is decoded/stored exactly once. This mattered
concretely: the real test asset's 20 sub-meshes all share one texture (the original Mecha-Ziu
exporter only ever wrote `Meshes[0]`'s texture name), so the original shape would have meant 20x
redundant decode/upload.

`gltf = { version = "1.4", default-features = false, features = ["utils", "names"] }` kept the
crate's `image`-decoding `import` feature out of `src/`'s dependency graph entirely, confirmed by
inspecting `cargo build`'s dependency list after adding it — no `image`/`jpeg`/`png` crate appears
transitively. `Gltf::open` (not `gltf::import`) plus a hand-rolled recursive node-transform
composition (the crate exposes only each node's *local* transform) worked exactly as the design
research predicted, with one correction made directly against the crate's real source rather than
taken on faith: `Transform::matrix()` returns glTF's column-major `[[f32;4];4]` layout, while this
engine's `Mat4::from_2d_array` is row-major, so `node_local_matrix` transposes on conversion — the
plan's Task 1 had incorrectly assumed no transpose was needed; verified against both the `gltf`
crate's own `Transform::decomposed` (reads translation from `m[3][0..3]`) and `m2s2-math`'s
`perspective`/`translation` constructor literals before writing the fix.

All 13 synthetic-fixture unit tests plus the real-asset regression test pass in plain `cargo test`
— this module touches no Vulkan handle and needs no `--ignored`/GPU driver gate, unlike the rest
of the renderer's GPU-dependent test suite. Every rejection test asserts on the specific error
message substring, not just `is_err()`, after discovering mid-implementation that several
fixtures could plausibly fail for an unrelated glTF schema-validation reason and still pass a bare
error check.

The real asset: a throwaway, uncommitted Python script (stdlib only — `struct`/`json`/`zlib`,
hand-rolling a TGA decoder and PNG encoder since Pillow wasn't available) parsed
`Mecha-Ziu/Game/Executable/Assets/Models/TestModel/MechaZiu.Model` using the exact binary layout
read directly from the original project's own `Model_Exporter.cpp`, remapped each of its 20
sub-meshes from the file's global vertex/triangle arrays to independent local vertex/index
buffers, transposed each sub-mesh's already-correct Maya-row-major world matrix into glTF's
column-major node-matrix convention, and converted the sibling `ziu.tga` to an embedded PNG. The
resulting `MechaZiu.glb` (550KB, committed under `examples/assets/mecha_ziu/`) round-trips through
`load_gltf` with vertex counts for all 20 sub-meshes (65, 395, 816, 348, 128, 50, 88, 187, 147×6,
59, 273, 81, 59, 273, 81 — totaling exactly 3785) matching the original file's own per-sub-mesh
`vcount` fields, independently cross-checked via a separate ad-hoc byte-level parse of the same
file earlier in the same session. Its texture decodes via `png::Decoder` to the original 512×512
resolution.

`examples/model.rs` builds clean and launches without error under a free-fly camera (reusing
`examples/cube.rs`'s input/camera scheme). Launching the compiled binary directly (rather than
via `cargo run`) needed `DYLD_LIBRARY_PATH=/usr/local/lib` for the Vulkan loader to resolve —
`cargo run`/`cargo test` apparently carry an equivalent path already.

**A real bug surfaced only by manual visual inspection** (this session has no display/screen-
recording access, so the user ran `cargo run --example model` and shared a screenshot): the mech
rendered as a small correctly-shaped/textured cluster (the sub-meshes whose stored transform
happens to be identity: pelvis, one torso piece, the gun-arm) surrounded by huge degenerate
sliver triangles (every other sub-mesh — head, limbs, both legs — all of which have a real,
non-identity transform). All 15 automated unit tests were green at the time; none of them
exercised an explicit glTF `"matrix"` node field with a *non-identity* value — every fixture
either used no transform or a TRS `"translation"` field, which the `gltf` crate converts to a
matrix internally via its own (correct) code path, never reaching this spec's
`node_local_matrix`/explicit-`"matrix"` handling at all. The converted asset's own node matrices
are the only place that path was actually exercised, and 17 of its 20 are non-identity — the gap
was real, not theoretical.

Root cause, confirmed by hand-simulating the exact committed code against one of the asset's real
matrices before touching anything: the (throwaway, uncommitted) converter's
`row_major_to_gltf_matrix` wrote `flat[col*4+row] = A[row][col]` (Maya's matrix `A`, read as-is)
instead of `flat[col*4+row] = A[col][row]` — the same expression with `row`/`col` swapped on the
right-hand side. Maya's `A` uses row-vector convention (`p' = p·A`, translation in row 3); glTF
and this engine use column-vector convention (`p' = M·p`, translation in column 3), and going
from one to the other requires an actual semantic transpose (`M = Aᵀ`), not just a storage-layout
repacking. The bug's result: translation values landed in matrix row 3 instead of column 3, where
they only ever reach a transformed vertex's **W** component — and only via a dot product with
that vertex's own local `(x,y,z)`, which differs per vertex. Different vertices in the same
mesh thus got different garbage W after the perspective divide, stretching triangles into exactly
the slivers in the screenshot; meshes with an identity matrix were unaffected since the bug is a
no-op on an identity matrix (confirmed directly: meshes 0-2 are the only three with an identity
transform in the real file, and are exactly the three that rendered correctly).

Fixed by swapping the two indices on the right-hand side (`A[col][row]`), verified numerically by
hand against mesh 4's real matrix before regenerating the asset (origin → `(4.4916, 1.2396,
5.2397)`, matching the file's own stored translation to 4 decimal places) and again after
regenerating by re-reading every one of the 20 meshes' `world_transform` back out through
`load_gltf` and confirming each matches the original file's row-3 translation values and has
`[0,0,0,1]` in its own row 3. Added
`explicit_matrix_field_places_translation_in_the_right_column_not_a_coincidence` to
`src/model.rs` to close the actual coverage gap — a loader-side test with a non-identity
`"matrix"` field, independent of the (still throwaway, still uncommitted) converter — so this
exact bug class can't reappear silently. The fixed asset's world-space bounding box shrank from a
scattered 32×22×18 to a compact 23×24×15, consistent with a properly assembled mech rather than
limbs flung to wherever a garbage transform happened to put them. `examples/model.rs`'s camera
framing constants were recalculated against the corrected bounding box. Not yet re-confirmed
visually by the user as of this writing — recommended before considering this fully closed.

Both Open Questions above were resolved during implementation: the converter emits a flat
structure (20 nodes, each a direct child of the scene, each carrying its own pre-composed world
matrix) rather than a real parent/child node chain — simpler and fully spec-compliant, since
`load_gltf`'s transform composition is correct either way. Asset lookup in both the regression
test and the example is anchored via `env!("CARGO_MANIFEST_DIR")`, not a bare relative path,
so it doesn't depend on the caller's working directory.
