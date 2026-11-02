// Vertex data types and GPU buffers. See specs/003-vertex-buffers.md.

use crate::math::Vec3;
use ash::vk;

/// A single vertex: a position and a color, both in clip space / RGB. `#[repr(C)]` + `Pod` so a
/// `&[Vertex]` can be uploaded to a GPU buffer byte-for-byte (see `Buffer`). `Vec3` is
/// `m2s2-math`'s `Vector3f32`, `Pod` under its `bytemuck` feature (identical layout to `[f32; 3]`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub color: Vec3,
}

impl Vertex {
    /// One vertex buffer, consumed one `Vertex` per vertex.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// Shader inputs: `location = 0` position, `location = 1` color — offsets derived from the
    /// struct so they can't drift from the field order.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, color) as u32),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_is_tightly_packed() {
        // position (Vec3 = 12 bytes) + color (Vec3 = 12 bytes), no padding.
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
    }

    #[test]
    fn binding_description_describes_one_per_vertex_buffer() {
        let b = Vertex::binding_description();
        assert_eq!(b.binding, 0);
        assert_eq!(b.stride, std::mem::size_of::<Vertex>() as u32);
        assert_eq!(b.input_rate, vk::VertexInputRate::VERTEX);
    }

    #[test]
    fn attribute_descriptions_map_position_then_color() {
        let a = Vertex::attribute_descriptions();
        // location 0 = position, at the start of the struct
        assert_eq!(a[0].location, 0);
        assert_eq!(a[0].binding, 0);
        assert_eq!(a[0].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(a[0].offset, 0);
        // location 1 = color, 12 bytes in (after the position Vec3)
        assert_eq!(a[1].location, 1);
        assert_eq!(a[1].binding, 0);
        assert_eq!(a[1].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(a[1].offset, 12);
    }
}
