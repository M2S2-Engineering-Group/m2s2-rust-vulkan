//! glTF model loading (spec 020): pure CPU-side parsing, no Vulkan dependency.

use crate::error::{Result, VulkanError};
use crate::math::{Mat4, Vec2, Vec3};
use crate::{Index, Vertex};
use gltf::{image::Source, Gltf, Image, Mesh, Node};
use std::collections::HashMap;
use std::path::Path;

pub struct LoadedModel {
    pub meshes: Vec<LoadedMesh>,
    pub textures: Vec<LoadedImage>,
}

pub struct LoadedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
    pub world_transform: Mat4,
    pub texture: Option<usize>,
}

pub struct LoadedImage {
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

/// Dedup state threaded through node traversal: the glTF document's own image index maps to
/// its position in `LoadedModel::textures`, so a shared texture is decoded/stored exactly once.
struct LoadContext<'a> {
    blob: &'a [u8],
    textures: Vec<LoadedImage>,
    image_index_to_texture: HashMap<usize, usize>,
}

pub fn load_gltf(path: &Path) -> Result<LoadedModel> {
    let gltf = Gltf::open(path)
        .map_err(|e| VulkanError::Other(format!("failed to open glTF file: {e}")))?;
    let blob = gltf.blob.as_deref().ok_or_else(|| {
        VulkanError::Other(
            "glTF file has no embedded binary chunk; only self-contained .glb is supported"
                .to_string(),
        )
    })?;

    let scene = gltf
        .document
        .default_scene()
        .or_else(|| gltf.document.scenes().next())
        .ok_or_else(|| VulkanError::Other("glTF document has no scenes".to_string()))?;

    let mut ctx = LoadContext {
        blob,
        textures: Vec::new(),
        image_index_to_texture: HashMap::new(),
    };
    let mut meshes = Vec::new();
    for node in scene.nodes() {
        visit_node(&node, Mat4::IDENTITY, &mut ctx, &mut meshes)?;
    }

    Ok(LoadedModel {
        meshes,
        textures: ctx.textures,
    })
}

fn visit_node(
    node: &Node,
    parent_world: Mat4,
    ctx: &mut LoadContext,
    meshes: &mut Vec<LoadedMesh>,
) -> Result<()> {
    let world = parent_world * node_local_matrix(node);

    if let Some(mesh) = node.mesh() {
        meshes.push(load_mesh(&mesh, world, ctx)?);
    }

    for child in node.children() {
        visit_node(&child, world, ctx, meshes)?;
    }

    Ok(())
}

/// `node.transform().matrix()` returns glTF's column-major `[[f32;4];4]` (outer index = column,
/// inner = row — confirmed against the `gltf` crate's own `Transform::decomposed`, which reads
/// translation from `m[3][0..3]`). This engine's `Mat4` is row-major (`from_2d_array`'s outer
/// index = row, matching `m2s2-math`'s `perspective`/`translation` constructors), so converting
/// requires a transpose, not a direct copy.
fn node_local_matrix(node: &Node) -> Mat4 {
    let m = node.transform().matrix();
    Mat4::from_2d_array([
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ])
}

fn load_mesh(mesh: &Mesh, world_transform: Mat4, ctx: &mut LoadContext) -> Result<LoadedMesh> {
    let mut primitives = mesh.primitives();
    let primitive = primitives
        .next()
        .ok_or_else(|| VulkanError::Other("mesh has no primitives".to_string()))?;
    if primitives.next().is_some() {
        return Err(VulkanError::Other(
            "mesh has more than one primitive; only exactly one is supported".to_string(),
        ));
    }

    let reader = primitive.reader(|_buffer| Some(ctx.blob));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| VulkanError::Other("primitive is missing POSITION".to_string()))?
        .collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .ok_or_else(|| VulkanError::Other("primitive is missing NORMAL".to_string()))?
        .collect();
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .ok_or_else(|| VulkanError::Other("primitive is missing TEXCOORD_0".to_string()))?
        .into_f32()
        .collect();

    if positions.len() != normals.len() || positions.len() != uvs.len() {
        return Err(VulkanError::Other(
            "primitive's POSITION/NORMAL/TEXCOORD_0 accessors have mismatched counts".to_string(),
        ));
    }
    if positions.len() > u16::MAX as usize {
        return Err(VulkanError::Other(format!(
            "primitive has {} vertices, exceeding the {} this engine's Index (u16) supports",
            positions.len(),
            u16::MAX
        )));
    }

    let vertices: Vec<Vertex> = positions
        .iter()
        .zip(normals.iter())
        .zip(uvs.iter())
        .map(|((p, n), uv)| Vertex {
            position: Vec3::new(p[0], p[1], p[2]),
            color: Vec3::new(1.0, 1.0, 1.0),
            uv: Vec2::new(uv[0], uv[1]),
            normal: Vec3::new(n[0], n[1], n[2]),
        })
        .collect();

    let indices: Vec<Index> = reader
        .read_indices()
        .ok_or_else(|| VulkanError::Other("primitive is missing indices".to_string()))?
        .into_u32()
        .map(|i| {
            if i as usize >= vertices.len() {
                Err(VulkanError::Other(format!(
                    "index {i} out of range for {} vertices",
                    vertices.len()
                )))
            } else if i > u16::MAX as u32 {
                Err(VulkanError::Other(format!("index {i} exceeds u16::MAX")))
            } else {
                Ok(i as Index)
            }
        })
        .collect::<Result<Vec<Index>>>()?;

    let texture = match primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
    {
        None => None,
        Some(info) => {
            let image = info.as_ref().source();
            Some(resolve_texture(&image, ctx)?)
        }
    };

    Ok(LoadedMesh {
        vertices,
        indices,
        world_transform,
        texture,
    })
}

fn resolve_texture(image: &Image, ctx: &mut LoadContext) -> Result<usize> {
    let image_index = image.index();
    if let Some(&existing) = ctx.image_index_to_texture.get(&image_index) {
        return Ok(existing);
    }

    let loaded = match image.source() {
        Source::View { view, mime_type } => {
            let start = view.offset();
            let end = start + view.length();
            LoadedImage {
                bytes: ctx.blob[start..end].to_vec(),
                mime_type: mime_type.to_string(),
            }
        }
        Source::Uri { .. } => {
            return Err(VulkanError::Other(
                "external image URIs are not supported; only .glb-embedded buffer views are"
                    .to_string(),
            ));
        }
    };

    let new_index = ctx.textures.len();
    ctx.textures.push(loaded);
    ctx.image_index_to_texture.insert(image_index, new_index);
    Ok(new_index)
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};
    use std::io::Write;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Tests in this module run in parallel within one process; every temp-file helper below
    /// draws from this counter so no two calls can ever collide on the same filename.
    static UNIQUE_FILE_COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Packs a glTF JSON document and a binary blob into valid GLB container bytes,
    /// per the glTF 2.0 binary container spec.
    fn glb_fixture(doc: Value, bin: &[u8]) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(&doc).unwrap();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(0x20); // space-pad JSON chunk
        }
        let mut bin_bytes = bin.to_vec();
        while !bin_bytes.len().is_multiple_of(4) {
            bin_bytes.push(0x00); // zero-pad BIN chunk
        }

        let json_chunk_len = json_bytes.len() as u32;
        let bin_chunk_len = bin_bytes.len() as u32;
        let total_len = 12 + 8 + json_chunk_len + 8 + bin_chunk_len;

        let mut out = Vec::new();
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&total_len.to_le_bytes());

        out.extend_from_slice(&json_chunk_len.to_le_bytes());
        out.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        out.extend_from_slice(&json_bytes);

        out.extend_from_slice(&bin_chunk_len.to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes());
        out.extend_from_slice(&bin_bytes);

        out
    }

    /// A minimal valid 1x1 RGBA PNG, built via the project's existing `png` dev-dependency
    /// rather than hand-typed byte literals.
    fn tiny_png() -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgba);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[255u8, 0, 0, 255]).unwrap();
        }
        bytes
    }

    /// Shared minimal triangle geometry (POSITION/NORMAL/TEXCOORD_0 + 3 u16 indices) used by
    /// tests that only care about node/mesh-count/transform behavior, not geometry content.
    /// Returns (bin bytes, the 4 accessors backing them, the 4 bufferViews backing those).
    fn shared_triangle_geometry() -> (Vec<u8>, Value, Value) {
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let normals_offset = bin.len();
        for _ in 0..3 {
            bin.extend_from_slice(&0.0f32.to_le_bytes());
            bin.extend_from_slice(&0.0f32.to_le_bytes());
            bin.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let uvs_offset = bin.len();
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let indices_offset = bin.len();
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let indices_len = bin.len() - indices_offset;

        let accessors = json!([
            {
                "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]
            },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
            { "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }
        ]);
        let buffer_views = json!([
            { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
            { "buffer": 0, "byteOffset": normals_offset, "byteLength": 36, "target": 34962 },
            { "buffer": 0, "byteOffset": uvs_offset, "byteLength": 24, "target": 34962 },
            { "buffer": 0, "byteOffset": indices_offset, "byteLength": indices_len, "target": 34963 }
        ]);

        (bin, accessors, buffer_views)
    }

    fn write_glb_to_tempfile(bytes: &[u8]) -> std::path::PathBuf {
        let unique = UNIQUE_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "m2s2_model_test_{}_{unique}.glb",
            std::process::id()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn single_textured_mesh_round_trips() {
        // Bin layout: 3 positions (f32x3 each, 36 bytes) @0, 3 normals @36, 3 uvs (f32x2, 24
        // bytes) @72, 3 u16 indices @96 (6 bytes), then the PNG bytes @104.
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let normals_offset = bin.len();
        for _ in 0..3 {
            bin.extend_from_slice(&0.0f32.to_le_bytes());
            bin.extend_from_slice(&0.0f32.to_le_bytes());
            bin.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let uvs_offset = bin.len();
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let indices_offset = bin.len();
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let indices_len = bin.len() - indices_offset;
        while bin.len() % 4 != 0 {
            bin.push(0); // pad before the image bytes so its bufferView offset is clean
        }
        let image_offset = bin.len();
        let png_bytes = tiny_png();
        let image_len = png_bytes.len();
        bin.extend_from_slice(&png_bytes);

        let doc = json!({
            "asset": { "version": "2.0" },
            "scenes": [ { "nodes": [0] } ],
            "scene": 0,
            "nodes": [ { "mesh": 0 } ],
            "meshes": [ {
                "primitives": [ {
                    "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                    "indices": 3,
                    "material": 0
                } ]
            } ],
            "materials": [ {
                "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } }
            } ],
            "textures": [ { "source": 0 } ],
            "images": [ { "bufferView": 4, "mimeType": "image/png" } ],
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                    "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]
                },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
                { "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
                { "buffer": 0, "byteOffset": normals_offset, "byteLength": 36, "target": 34962 },
                { "buffer": 0, "byteOffset": uvs_offset, "byteLength": 24, "target": 34962 },
                { "buffer": 0, "byteOffset": indices_offset, "byteLength": indices_len, "target": 34963 },
                { "buffer": 0, "byteOffset": image_offset, "byteLength": image_len }
            ],
            "buffers": [ { "byteLength": bin.len() } ]
        });

        let glb = glb_fixture(doc, &bin);
        let path = write_glb_to_tempfile(&glb);

        let model = super::load_gltf(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(model.meshes.len(), 1);
        assert_eq!(model.textures.len(), 1);
        assert_eq!(model.textures[0].mime_type, "image/png");
        assert_eq!(model.meshes[0].texture, Some(0));
        assert_eq!(model.meshes[0].vertices.len(), 3);
        assert_eq!(model.meshes[0].indices, vec![0, 1, 2]);
        assert_eq!(model.meshes[0].world_transform, crate::math::Mat4::IDENTITY);
        assert!(model.meshes[0]
            .vertices
            .iter()
            .all(|v| v.color == crate::math::Vec3::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn non_glb_without_blob_is_rejected() {
        // A bare JSON-only document, not wrapped in a GLB container at all — Gltf::open sniffs
        // the magic bytes and parses it as plain glTF JSON with no embedded binary chunk.
        let doc = json!({
            "asset": { "version": "2.0" },
            "scenes": [ { "nodes": [] } ],
            "scene": 0
        });
        let unique = UNIQUE_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "m2s2_model_test_nogltf_{}_{unique}.gltf",
            std::process::id()
        ));
        std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();

        let result = super::load_gltf(&path);
        std::fs::remove_file(&path).ok();

        assert!(result.is_err());
    }

    #[test]
    fn missing_texture_is_none_not_an_error() {
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let normals_offset = bin.len();
        for _ in 0..3 {
            bin.extend_from_slice(&0.0f32.to_le_bytes());
            bin.extend_from_slice(&0.0f32.to_le_bytes());
            bin.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let uvs_offset = bin.len();
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let indices_offset = bin.len();
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let indices_len = bin.len() - indices_offset;

        let doc = json!({
            "asset": { "version": "2.0" },
            "scenes": [ { "nodes": [0] } ],
            "scene": 0,
            "nodes": [ { "mesh": 0 } ],
            "meshes": [ {
                "primitives": [ {
                    "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                    "indices": 3
                } ]
            } ],
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                    "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]
                },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
                { "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
                { "buffer": 0, "byteOffset": normals_offset, "byteLength": 36, "target": 34962 },
                { "buffer": 0, "byteOffset": uvs_offset, "byteLength": 24, "target": 34962 },
                { "buffer": 0, "byteOffset": indices_offset, "byteLength": indices_len, "target": 34963 }
            ],
            "buffers": [ { "byteLength": bin.len() } ]
        });

        let glb = glb_fixture(doc, &bin);
        let path = write_glb_to_tempfile(&glb);

        let model = super::load_gltf(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(model.meshes.len(), 1);
        assert_eq!(model.meshes[0].texture, None);
        assert!(model.textures.is_empty());
    }

    #[test]
    fn world_transform_composes_through_a_three_level_chain_with_meshes_at_every_level() {
        // Node A (translate (1,0,0), mesh 0) -> child Node B (translate (0,2,0), mesh 1) ->
        // child Node C (translate (0,0,3), mesh 2). All three meshes reuse the same shared
        // geometry/accessors; only node structure and transforms are under test here.
        let (bin, accessors, buffer_views) = shared_triangle_geometry();
        let bin_len = bin.len();

        let doc = json!({
            "asset": { "version": "2.0" },
            "scenes": [ { "nodes": [0] } ],
            "scene": 0,
            "nodes": [
                { "mesh": 0, "translation": [1.0, 0.0, 0.0], "children": [1] },
                { "mesh": 1, "translation": [0.0, 2.0, 0.0], "children": [2] },
                { "mesh": 2, "translation": [0.0, 0.0, 3.0] }
            ],
            "meshes": [
                { "primitives": [ { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3 } ] },
                { "primitives": [ { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3 } ] },
                { "primitives": [ { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3 } ] }
            ],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [ { "byteLength": bin_len } ]
        });

        let glb = glb_fixture(doc, &bin);
        let path = write_glb_to_tempfile(&glb);
        let model = super::load_gltf(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(model.meshes.len(), 3); // meshes at every level of the chain count, not just leaves
        let translation_of = |m: &super::LoadedMesh| {
            (
                m.world_transform.get(0, 3).unwrap(),
                m.world_transform.get(1, 3).unwrap(),
                m.world_transform.get(2, 3).unwrap(),
            )
        };
        assert_eq!(translation_of(&model.meshes[0]), (1.0, 0.0, 0.0));
        assert_eq!(translation_of(&model.meshes[1]), (1.0, 2.0, 0.0));
        assert_eq!(translation_of(&model.meshes[2]), (1.0, 2.0, 3.0));
    }

    #[test]
    fn two_scenes_with_no_explicit_default_uses_the_first() {
        let (bin, accessors, buffer_views) = shared_triangle_geometry();
        let bin_len = bin.len();

        let doc = json!({
            "asset": { "version": "2.0" },
            "scenes": [ { "nodes": [0] }, { "nodes": [1] } ],
            "nodes": [
                { "mesh": 0 },
                { "mesh": 1 }
            ],
            "meshes": [
                { "primitives": [ { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3 } ] },
                { "primitives": [ { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3 } ] }
            ],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [ { "byteLength": bin_len } ]
        });

        let glb = glb_fixture(doc, &bin);
        let path = write_glb_to_tempfile(&glb);
        let model = super::load_gltf(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // No top-level "scene" index: falls back to the first scene (index 0), which contains
        // node 0 (mesh 0) only — not node 1's scene.
        assert_eq!(model.meshes.len(), 1);
    }
}
