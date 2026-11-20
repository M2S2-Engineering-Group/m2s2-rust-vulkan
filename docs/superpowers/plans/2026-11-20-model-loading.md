# Model loading (glTF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-Rust, Vulkan-free glTF loader (`src/model.rs`) that turns a self-contained `.glb` into plain geometry/transform/texture-reference data the existing `create_mesh`/`create_texture`/`Object` APIs already consume, and prove it against a real converted asset (a 20-part mech model) in a new windowed example.

**Architecture:** One new standalone module, `src/model.rs` (public, alongside `src/math.rs` — no Vulkan dependency, unit-testable in CI without a GPU). It depends only on the `gltf` crate's non-`import` surface (`Gltf::open`, `Document`, `mesh::Reader`) plus hand-rolled recursive node-transform composition. A one-off, uncommitted script converts the legacy Mecha-Ziu `.Model` asset into the one `.glb` this spec ships; a new `examples/model.rs` exercises the whole path end-to-end under the existing free-fly camera.

**Tech Stack:** Rust, `gltf` crate (read-only, `import` feature disabled), `serde_json` (test-only, for building GLB fixtures), existing `m2s2-math`/`Vertex`/`Index`/`Object` types.

**Spec:** [specs/020-model-loading.md](../../../specs/020-model-loading.md)

## Global Constraints

- `Cargo.toml` `[dependencies]`: `gltf = { version = "1.4", default-features = false, features = ["utils", "names"] }` — never the crate's default `import` feature (it unconditionally pulls in the `image` crate).
- `Index = u16` (from `src/renderer/buffer.rs`): any primitive whose `POSITION` accessor count exceeds `u16::MAX` (65535) is rejected with `Err`, not truncated.
- Only `Source::View` glTF images are supported (buffer-view-embedded). `Source::Uri` (external or base64) is rejected with `Err`.
- Exactly one primitive per mesh is supported. A mesh with more than one primitive is rejected with `Err`.
- Every primitive must have `POSITION`, `NORMAL`, and `TEXCOORD_0` accessors. Any missing is rejected with `Err`.
- No per-vertex `COLOR_0` support: every loaded vertex's `color` field is `Vec3::new(1.0, 1.0, 1.0)`.
- The module touches no Vulkan handles and has no `VulkanDevice`/`VulkanRenderer` dependency — every test in this plan runs in plain `cargo test` (no `--ignored`, no GPU).
- `LoadedModel::textures` holds each distinct glTF image **once**, referenced by index from `LoadedMesh::texture: Option<usize>` — never duplicated per mesh.
- Only `.glb` (self-contained, embedded binary chunk via `Gltf::open`'s `blob`) is supported. A `.gltf` with no blob is rejected with `Err`.

## Review Focus

- A mesh attached to an intermediate (non-leaf) node, not just a leaf — the real Mecha-Ziu asset has meshes at every level of its hierarchy (pelvis itself carries a mesh, and so do its descendants), and a loader that only walks leaves would silently drop most of the model.
- A `.glb` with more than one `scene`, or no explicit `scene` index set — the spec never pins a rule; a reasonable person expects the file's own default scene to render, not an arbitrary one or a silent no-op.
- An index value that references a vertex position beyond what that primitive's attribute accessors actually contain (a malformed or hand-edited file) — a reasonable person expects a clear `Err`, not a panic from an out-of-bounds slice index.
- Two primitive attribute accessors with mismatched element counts (e.g. 10 positions but 8 normals) — naively zipping iterators of different lengths silently truncates to the shorter one instead of surfacing the corrupt file.
- A `.gltf` file (JSON + separate `.bin`, no embedded blob) handed to `load_gltf` — since only self-contained `.glb` is supported, this must be a clear `Err`, not a panic unwrapping `gltf.blob`.

---

## Task 1: Core loader — single mesh, texture, and the two baseline rejections

**Files:**
- Create: `src/model.rs`
- Modify: `src/lib.rs` (add `pub mod model;`)
- Modify: `Cargo.toml` (add the `gltf` dependency under `[dependencies]`; add `serde_json = "1"` under `[dev-dependencies]`)
- Test: inline `#[cfg(test)] mod tests` in `src/model.rs`

**Interfaces:**
- Produces (used by every later task in this plan):
  - `pub struct LoadedModel { pub meshes: Vec<LoadedMesh>, pub textures: Vec<LoadedImage> }`
  - `pub struct LoadedMesh { pub vertices: Vec<Vertex>, pub indices: Vec<Index>, pub world_transform: Mat4, pub texture: Option<usize> }`
  - `pub struct LoadedImage { pub bytes: Vec<u8>, pub mime_type: String }`
  - `pub fn load_gltf(path: &std::path::Path) -> Result<LoadedModel>`
  - `fn glb_fixture(doc: serde_json::Value, bin: &[u8]) -> Vec<u8>` (test-only, `#[cfg(test)]`) — packs a glTF JSON document and a binary blob into valid GLB container bytes. Exact algorithm (fixed by the [glTF 2.0 binary spec](https://docs.vulkan.org) — no judgment call, implement precisely):
    - 12-byte header: `b"glTF"` (magic, little-endian u32 `0x46546C67`), `2u32` (version), `total_length: u32` (computed last, write after the rest is known).
    - JSON chunk: `chunk_length: u32` = the JSON text's byte length rounded up to a multiple of 4 (pad with ASCII space `0x20`), `chunk_type: u32` = `0x4E4F534A`, then the (padded) JSON bytes.
    - BIN chunk: `chunk_length: u32` = `bin.len()` rounded up to a multiple of 4 (pad with `0x00`), `chunk_type: u32` = `0x004E4942`, then the (padded) bin bytes.
    - `total_length` = `12 + 8 + json_chunk_len + 8 + bin_chunk_len`.

- [ ] **Step 1: Add the dependency**

Add to `Cargo.toml`:
```toml
gltf = { version = "1.4", default-features = false, features = ["utils", "names"] }
```
and to `[dev-dependencies]`:
```toml
serde_json = "1"
```
Run `cargo build` to confirm it resolves and compiles with no other code changes yet.

- [ ] **Step 2: Write the failing test — single node, mesh, embedded PNG texture**

In `src/model.rs`, write `glb_fixture` as specified above, then:

```rust
#[test]
fn single_textured_mesh_round_trips() {
    // One node (identity transform) with one mesh/primitive: a 3-vertex triangle with
    // POSITION/NORMAL/TEXCOORD_0, UNSIGNED_SHORT indices [0,1,2], a material whose
    // baseColorTexture points at images[0] (bufferView-embedded, mimeType "image/png",
    // bytes = any valid 1x1 PNG you embed as a byte literal).
    let model = load_gltf(&fixture_path).unwrap(); // write the .glb to a tempfile; load_gltf takes a Path
    assert_eq!(model.meshes.len(), 1);
    assert_eq!(model.textures.len(), 1);
    assert_eq!(model.textures[0].mime_type, "image/png");
    assert_eq!(model.meshes[0].texture, Some(0));
    assert_eq!(model.meshes[0].vertices.len(), 3);
    assert_eq!(model.meshes[0].indices, vec![0, 1, 2]);
    assert_eq!(model.meshes[0].world_transform, Mat4::IDENTITY);
    // every loaded vertex defaults to white — no COLOR_0 support (Global Constraints)
    assert!(model.meshes[0].vertices.iter().all(|v| v.color == Vec3::new(1.0, 1.0, 1.0)));
}
```

Build the JSON `serde_json::json!{...}` document by hand: one `scene` (default), one `node` (mesh index 0, no `matrix` field = identity), one `mesh` with one `primitive` (`attributes`: POSITION/NORMAL/TEXCOORD_0 accessor indices, `indices`: accessor index, `material`: 0), one `material` with `pbrMetallicRoughness.baseColorTexture.index = 0`, one `texture` (`source: 0`), one `image` (`bufferView: N`, `mimeType: "image/png"`), the `accessors`/`bufferViews`/`buffers` backing all of the above, with the PNG bytes and vertex/index data laid out at known offsets in the `bin` byte vector you pass to `glb_fixture`. Use `bufferView` targets `34962` (vertex data) and `34963` (indices) — optional but write them for clarity.

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib model::tests::single_textured_mesh_round_trips -- --nocapture`
Expected: FAIL (`load_gltf` undefined, or compile error — `LoadedModel`/`LoadedMesh`/`LoadedImage` not yet defined).

- [ ] **Step 4: Implement `load_gltf`**

- `Gltf::open(path)` → `Err` if it fails (wrap in this crate's `VulkanError`/`Result` — check `src/error.rs` for the right variant or add one, e.g. `VulkanError::Model(String)`, matching the existing error-enum pattern).
- If `gltf.blob` is `None` → `Err` (Global Constraints: `.gltf`-with-external-bin is rejected here, first).
- Resolve the scene to walk: `document.default_scene()`, or else `document.scenes().next()` (Review Focus: scene-selection rule).
- Recursively walk every node in that scene (not just roots — every node reachable via `node.children()`, at every depth), tracking each node's locally-composed world transform (`parent_world * node.transform().matrix()`, both as column-major `Mat4`; the crate's `transform().matrix()` already returns glTF's column-major 16-`f32` layout, so no additional transpose is needed in Rust code — only the separate conversion script needs one, see Task 5). For each node that has a mesh (regardless of depth — Review Focus item 1), process it as described in Steps 5–6 below, using the composed world transform at that node.
- For each such node's mesh: if `mesh.primitives().count() != 1` → `Err`. Otherwise, build a `mesh::Reader` backed by a closure `|buffer: gltf::Buffer| gltf.blob.as_deref()` (every buffer in a self-contained `.glb` resolves to the one blob). Read `positions`, `normals`, `tex_coords(0)`; any `None` → `Err` (Global Constraints). Collect into `Vec<Vertex>` (`color` always white) — `positions.len()` must equal `normals.len()` and `tex_coords.len()` (Review Focus item 4: mismatched counts → `Err`, not a truncating zip). If `vertices.len() > u16::MAX as usize` → `Err` (Global Constraints).
- Read indices: `reader.read_indices()` → `Err` if `None`; `.into_u32()` then cast each to `u16`, erroring (not panicking/truncating) if any value exceeds `u16::MAX`, and erroring if any index is `>= vertices.len()` (Review Focus item 3).
- Texture: `primitive.material().pbr_metallic_roughness().base_color_texture()`; if `None`, mesh's `texture` field is `None`. If present, resolve its `texture().source()` (a `gltf::Image`) and look it up by **glTF image index** in a `HashMap<usize, usize>` you build once per `load_gltf` call mapping "glTF image index" → "position already pushed into the `LoadedModel::textures` Vec" — push a new `LoadedImage` only the first time an image index is seen (this is the dedup). The image's `Source` must be `Source::View { view, mime_type }` (else `Err`, per Global Constraints); slice `view`'s byte range out of `gltf.blob`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib model::tests::single_textured_mesh_round_trips`
Expected: PASS.

- [ ] **Step 6: Write and pass the two baseline-rejection tests**

```rust
#[test]
fn non_glb_without_blob_is_rejected() {
    // Write a bare .gltf-style JSON-only file (no GLB container at all, or a GLB whose
    // BIN chunk is omitted) and assert load_gltf(&path).is_err().
}

#[test]
fn missing_texture_is_none_not_an_error() {
    // Same fixture as Step 2's primitive/mesh, but the primitive has no `material` field
    // at all. Assert load_gltf(..).unwrap().meshes[0].texture is None and textures is empty.
}
```
Run `cargo test --lib model::` and confirm both pass alongside Step 2's test.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/model.rs src/lib.rs src/error.rs
git commit -m "feat(model): add glTF loader for single textured meshes"
```

---

## Task 2: Node-hierarchy world-transform composition

**Files:**
- Modify: `src/model.rs`

**Interfaces:**
- Consumes: Task 1's `load_gltf`, `LoadedModel`/`LoadedMesh` (transform composition already exists as internal logic from Task 1, Step 4 — this task's tests are what actually prove it, since Task 1's fixture only used an identity-transform single node).

- [ ] **Step 1: Write the failing test — three-level parent chain, mesh at every level**

```rust
#[test]
fn world_transform_composes_through_a_three_level_chain_with_meshes_at_every_level() {
    // Node A (translate by (1,0,0), has its own mesh) -> child Node B (translate by (0,2,0),
    // has its own mesh) -> child Node C (translate by (0,0,3), has its own mesh).
    // Use `matrix` fields (16 floats, column-major, translation in the last 4) rather than
    // TRS, since that's what the real conversion script will emit (Task 5).
    let model = load_gltf(&fixture_path).unwrap();
    assert_eq!(model.meshes.len(), 3); // Review Focus: meshes at non-leaf nodes count too
    // find each mesh by a distinguishing vertex count or order of insertion (document traversal
    // order is scene.nodes() depth-first, so index 0/1/2 correspond to A/B/C respectively)
    assert_eq!(model.meshes[0].world_transform.w_axis.truncate(), Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(model.meshes[1].world_transform.w_axis.truncate(), Vec3::new(1.0, 2.0, 0.0));
    assert_eq!(model.meshes[2].world_transform.w_axis.truncate(), Vec3::new(1.0, 2.0, 3.0));
}
```
(Adjust the exact `Mat4`/`Vec3` accessor names to whatever `m2s2-math` actually exposes for reading a translation column — check `src/math.rs` if `w_axis`/`truncate()` isn't right; the intent is "read the translation component of the composed matrix.")

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib model::tests::world_transform_composes_through_a_three_level_chain_with_meshes_at_every_level`
Expected: FAIL only if Task 1's traversal was accidentally leaves-only or roots-only; if Task 1 already walks every node recursively composing transforms correctly, this may pass immediately — that's fine, it still pins the behavior. Confirm it currently reports PASS or FAIL and proceed accordingly (no code change needed if it already passes; otherwise fix the traversal in `src/model.rs` from Task 1, Step 4 to recurse into every node regardless of whether it carries a mesh).

- [ ] **Step 3: Run test to verify it passes**

Run: `cargo test --lib model::tests::world_transform_composes_through_a_three_level_chain_with_meshes_at_every_level`
Expected: PASS.

- [ ] **Step 4: Write and pass the multi-scene default test**

```rust
#[test]
fn two_scenes_with_no_explicit_default_uses_the_first() {
    // A document with `scenes: [sceneA, sceneB]` and no top-level `scene` index.
    // sceneA has one mesh-bearing node; sceneB has a different one (e.g. a different
    // vertex count) so the test can tell which was used.
    let model = load_gltf(&fixture_path).unwrap();
    assert_eq!(model.meshes.len(), 1);
    // assert it's sceneA's mesh specifically, by its distinguishing vertex count
}
```
Run `cargo test --lib model::` and confirm it passes alongside the chain test.

- [ ] **Step 5: Commit**

```bash
git add src/model.rs
git commit -m "test(model): pin world-transform composition through node hierarchies"
```

---

## Task 3: Texture dedup and remaining rejection paths

**Files:**
- Modify: `src/model.rs`

**Interfaces:**
- Consumes: Task 1's image-index `HashMap` dedup logic (already implemented in Task 1, Step 4) — this task's first test is what actually proves it with more than one mesh.

- [ ] **Step 1: Write the failing test — two meshes share one image**

```rust
#[test]
fn two_meshes_sharing_one_image_dedupe_into_one_texture() {
    // Two sibling nodes, each with its own mesh/primitive, both materials' baseColorTexture
    // pointing at the SAME images[0] entry.
    let model = load_gltf(&fixture_path).unwrap();
    assert_eq!(model.textures.len(), 1);
    assert_eq!(model.meshes[0].texture, Some(0));
    assert_eq!(model.meshes[1].texture, Some(0));
}
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cargo test --lib model::tests::two_meshes_sharing_one_image_dedupe_into_one_texture`
Expected: PASS if Task 1's `HashMap<usize, usize>` dedup (Task 1 Step 4) was implemented correctly; if it instead pushes a new `LoadedImage` per mesh, this fails — fix `load_gltf` to key the push on the glTF image index, not call-site order.

- [ ] **Step 3: Write and pass the five remaining rejection-path tests**

```rust
#[test]
fn multi_primitive_mesh_is_rejected() { /* mesh.primitives has 2 entries */ }

#[test]
fn primitive_missing_normal_is_rejected() { /* attributes has POSITION+TEXCOORD_0, no NORMAL */ }

#[test]
fn primitive_missing_texcoord_is_rejected() { /* attributes has POSITION+NORMAL, no TEXCOORD_0 */ }

#[test]
fn primitive_over_65535_vertices_is_rejected() {
    // POSITION accessor count = 65536, backed by 65536 * 3 zeroed f32s in bin.
}

#[test]
fn external_uri_image_is_rejected() {
    // images[0] = { "uri": "foo.png" } instead of bufferView/mimeType.
}

#[test]
fn out_of_range_index_is_rejected_not_panicking() {
    // indices accessor contains a value >= the primitive's vertex count.
}

#[test]
fn mismatched_attribute_accessor_counts_is_rejected() {
    // POSITION accessor count = 10, NORMAL accessor count = 8.
}
```
Run `cargo test --lib model::` after each and confirm all pass without panicking (a panic means `assert!(result.is_err())`-style tests actually crashed the test binary instead of returning cleanly — check for that specifically on the out-of-range-index case).

- [ ] **Step 4: Run the full module's test suite**

Run: `cargo test --lib model::`
Expected: every test in `src/model.rs` passes (Tasks 1–3 combined).

- [ ] **Step 5: Commit**

```bash
git add src/model.rs
git commit -m "feat(model): dedupe shared textures; reject malformed/unsupported glTF inputs"
```

---

## Task 4: Public re-export, quality gates, and standalone unit-test gate check

**Files:**
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: Task 1's `LoadedModel`/`LoadedMesh`/`LoadedImage`/`load_gltf` (already `pub` inside `model`, via `pub mod model;` from Task 1 Step 1 — this task confirms the crate-level build/lint/test gates all pass with the new module in place, matching `math.rs`'s existing path-accessed style, i.e. callers use `m2s2_rust_vulkan::model::load_gltf`, no flat crate-root re-export needed).

- [ ] **Step 1: Run the full quality gate suite**

Run, in order:
```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets
cargo test --all-targets
```
Expected: all four exit 0. Fix any warning/failure in `src/model.rs` before moving on — do not suppress with `#[allow(...)]` unless the existing codebase already does so for an equivalent pattern elsewhere.

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "chore(model): close out quality gates for the glTF loader module"
```
(Skip this commit if Step 1 required no changes.)

---

## Task 5: Convert the Mecha-Ziu mech to `MechaZiu.glb`

**Files:**
- Create (not committed — deleted at the end of this task): a throwaway conversion script, any location/language outside the repo's normal build (e.g. a scratch Python script).
- Create (committed): `examples/assets/mecha_ziu/MechaZiu.glb`

**Interfaces:**
- Consumes: nothing from this crate — this task only produces a file Task 6 will load with Task 1–3's `load_gltf`.

- [ ] **Step 1: Write the throwaway converter**

Source data: `/Users/mgmaster24/projects/Mecha-Ziu/Game/Executable/Assets/Models/TestModel/MechaZiu.Model` and its sibling `ziu.tga`. `MechaZiu.Model`'s binary layout (confirmed against `Mecha-Ziu/Model_Exporter/Model_Exporter/ModelExporter.cpp`'s `WriteModelData_Binary`):

```
texture_filename: null-terminated string       ("ziu.tga")
mesh_count, vert_count, tri_count: u32 each     (little-endian)
repeated vert_count times:
    pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, u, v: f32 each (8 floats, 32 bytes/vertex)
repeated tri_count times:
    i0, i1, i2: u32 each                        (indices into the GLOBAL vertex array above)
repeated mesh_count times:
    name, transform_name, parent_name: null-terminated strings
    vert_count, tri_count, start_index, end_index: u32 each
      (start_index/end_index are an INCLUSIVE range into the GLOBAL triangle array above —
       this mesh's triangles are tris[start_index..=end_index])
    world_matrix: 16 x f32                      (Maya MMatrix, ROW-MAJOR, row-vector convention:
                                                  translation is in matrix[3][0..3])
(remaining animation-keyframe bytes: ignored per spec's non-goals)
```

Algorithm:
1. Parse the header and the two GLOBAL arrays (vertices, triangles) exactly as laid out above.
2. For each of the `mesh_count` sub-meshes: slice `tris[start_index..=end_index]`; collect the **set** of distinct global vertex indices these triangles reference (preserving first-seen order); build this sub-mesh's own local vertex array from just those, and remap each triangle's three global indices to local 0-based indices into that new array. (This remap is required — the spec's engine-side loader expects one self-contained vertex/index buffer per mesh, not a view into the global arrays.)
3. Convert each sub-mesh's `world_matrix` from Maya's row-major layout to glTF's column-major `node.matrix` layout: `gltf_matrix[c*4+r] = maya_matrix[r*4+c]` (a transpose) — this is the one real coordinate-system conversion step; no axis flip is needed (Maya and glTF are both right-handed, Y-up).
4. Decode `ziu.tga` (uncompressed 512×512 RGBA Targa — the `image` crate's `tga` feature, or any TGA reader, handles this) and re-encode it as PNG bytes.
5. Emit one `.glb`: one glTF `buffer` holding (a) each sub-mesh's local vertex data (POSITION/NORMAL/TEXCOORD_0, as three separate accessors backed by non-interleaved or interleaved layout — implementer's choice) and index data, plus (b) the PNG bytes from Step 4 in one more buffer view; one `material` with `baseColorTexture` pointing at that one embedded image, referenced by every sub-mesh's primitive; `mesh_count` (20) nodes, each with `matrix` set to its Step 3 result and referencing its own mesh index — all as direct children of the scene (a flat structure is sufions; the `parent_name` strings read in Step 1 are not otherwise used, since the already-resolved world matrices make real glTF node nesting unnecessary — see spec's Open Questions).
6. Write the result to `examples/assets/mecha_ziu/MechaZiu.glb`.

- [ ] **Step 2: Run the converter**

Run the script once against the real source files. Confirm `examples/assets/mecha_ziu/MechaZiu.glb` exists and is non-empty.

- [ ] **Step 3: Write and pass a loader regression test against the real asset**

In `src/model.rs`:
```rust
#[test]
fn mecha_ziu_glb_loads_all_twenty_meshes_and_one_shared_texture() {
    // CARGO_MANIFEST_DIR anchors this to the crate root regardless of the CWD `cargo test`
    // happens to be invoked from (resolves the spec's second Open Question).
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/mecha_ziu/MechaZiu.glb");
    let model = load_gltf(&path).unwrap();
    assert_eq!(model.meshes.len(), 20);
    assert_eq!(model.textures.len(), 1); // every sub-mesh shares the one ziu.tga-derived PNG
    assert!(model.meshes.iter().all(|m| m.texture == Some(0)));
    let total_vertices: usize = model.meshes.iter().map(|m| m.vertices.len()).sum();
    assert!(total_vertices > 0 && total_vertices <= 3785); // <= the original master vert count; the
    // per-submesh remap in Step 1 above can only reduce this (shared verts across triangles within
    // one submesh get deduped into fewer unique local vertices), never increase it
}
```
Run: `cargo test --lib model::tests::mecha_ziu_glb_loads_all_twenty_meshes_and_one_shared_texture`
Expected: PASS. If it fails, the bug is in the converter script from Step 1, not `load_gltf` (which Tasks 1–3 already proved correct against synthetic fixtures) — fix the script and re-run Step 2.

- [ ] **Step 4: Delete the converter script and commit the asset**

```bash
rm /path/to/the/throwaway/script.py
git add examples/assets/mecha_ziu/MechaZiu.glb src/model.rs
git commit -m "test(model): add real-asset regression test against the converted Mecha-Ziu mech"
```

---

## Task 6: `examples/model.rs` — windowed manual verification

**Files:**
- Create: `examples/model.rs`
- Modify: `Cargo.toml` (add the `[[example]]` entry: `name = "model"`, `path = "examples/model.rs"`)

**Interfaces:**
- Consumes: `m2s2_rust_vulkan::model::{load_gltf, LoadedModel}` (Task 1), `VulkanRenderer::{with_window, create_mesh, create_texture, render_objects}` and `Object`/`SpecularMaterial` (all pre-existing), `png::Decoder` (existing dev-dependency, used the same way `examples/cube.rs`'s texture-decoding would if it decoded files — check `examples/cube.rs` for its camera/input/event-loop scaffolding to mirror, since this example reuses that structure wholesale, swapping only the geometry source).

- [ ] **Step 1: Write the example**

Structure, mirroring `examples/cube.rs`'s window/event-loop/free-fly-camera scaffolding:
1. Create the window and `VulkanRenderer::with_window(...)` exactly as `cube.rs` does.
2. `let asset_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/mecha_ziu/MechaZiu.glb");` then `let model: LoadedModel = m2s2_rust_vulkan::model::load_gltf(&asset_path)?;` — anchoring to `CARGO_MANIFEST_DIR` rather than a bare relative path resolves the spec's second Open Question (asset lookup must not depend on the caller's CWD).
3. For each `model.textures[i]`: decode `bytes` via `png::Decoder` into RGBA8, call `renderer.create_texture(...)`, collect into `Vec<TextureId>`.
4. For each `model.meshes[i]`: `renderer.create_mesh(&m.vertices, &m.indices)?`, then build `Object { mesh, model: m.world_transform, texture: m.texture.map(|t| texture_ids[t]), specular: SpecularMaterial::default() }`.
5. Reuse `cube.rs`'s free-fly camera/input/render-loop code unchanged, rendering the fixed `Vec<Object>` built in Step 4 every frame (no per-frame animation needed — this is a static-model demo).

- [ ] **Step 2: Build and run it**

Run: `cargo build --example model`
Expected: compiles clean.

Run: `cargo run --example model`
Expected: a window opens showing the textured mech reassembled correctly (pelvis/torso/gun-arm/head/legs all in their modeled positions, not scattered at the origin) under the free-fly camera. This is the spec's manual-verification step — visually confirm it, then close the window.

- [ ] **Step 3: Commit**

```bash
git add examples/model.rs Cargo.toml
git commit -m "feat(examples): add model.rs rendering the converted Mecha-Ziu mech via load_gltf"
```

---

## Task 7: Spec Outcome, ROADMAP, and final gate pass

**Files:**
- Modify: `specs/020-model-loading.md` (fill in `## Outcome`)
- Modify: `ROADMAP.md` (check off the Phase 3 "Model loading" bullet)

- [ ] **Step 1: Fill in the spec's Outcome section**

Summarize what shipped vs. the draft: the `gltf = {default-features = false, features = ["utils","names"]}` dependency choice and why; the `LoadedModel`/`textures` dedup shape (and that it was added mid-design via the builder-check review, not in the first draft); the real vertex/mesh counts produced for `MechaZiu.glb`; confirmation that all quality gates and the full `cargo test --all-targets` (no `--ignored` needed — this module requires no GPU) pass.

- [ ] **Step 2: Check off ROADMAP.md**

Change:
```
- [ ] **Model loading**: glTF (preferred — PBR-friendly material model) and/or OBJ
```
to:
```
- [x] **Model loading**: glTF, geometry/transform/texture-reference only (no scene graph, no
      material system, no animation) — [spec 020](specs/020-model-loading.md). OBJ remains a
      future option per the roadmap's "and/or".
```

- [ ] **Step 3: Run the full quality gate suite one last time**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets
cargo test --all-targets
```
Expected: all four exit 0.

- [ ] **Step 4: Commit**

```bash
git add specs/020-model-loading.md ROADMAP.md
git commit -m "docs(spec): record spec 020 outcome; check off model loading in ROADMAP"
```
