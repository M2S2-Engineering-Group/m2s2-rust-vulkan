//! The caller-facing scene object: one drawn copy of a renderer-owned mesh.
use super::{
    mesh::MeshId,
    texture::{SpecularMaterial, TextureId},
};
use crate::math::Mat4;

/// One drawn copy of one of the renderer's uploaded meshes (identified by `mesh`). `texture: None`
/// samples the renderer's 1x1 white default, so an untextured object reproduces the pre-texture look
/// (`white * vertexColor == vertexColor`).
pub struct Object {
    pub model: Mat4,
    pub mesh: MeshId,
    pub texture: Option<TextureId>,
    pub specular: SpecularMaterial,
}
