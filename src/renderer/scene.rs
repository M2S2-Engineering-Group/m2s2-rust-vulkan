//! The caller-facing scene object: one drawn copy of a renderer-owned mesh.
use super::{material::Material, mesh::MeshId};
use crate::math::Mat4;

/// One drawn copy of one of the renderer's uploaded meshes (identified by `mesh`). `material`
/// bundles what the shader needs (texture, base-color tint, specular) — see spec 021.
/// `Material::default()`'s `texture: None` samples the renderer's 1x1 white default, so an
/// untextured, untinted object reproduces the pre-material-system look
/// (`white * vertexColor == vertexColor`).
pub struct Object {
    pub model: Mat4,
    pub mesh: MeshId,
    pub material: Material,
}
