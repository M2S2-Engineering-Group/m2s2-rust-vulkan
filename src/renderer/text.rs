//! Screen-space bitmap text. Coordinates are physical pixels from the top-left.
use super::{
    allocator::Allocator,
    buffer::Buffer,
    command::FRAMES_IN_FLIGHT,
    device::VulkanDevice,
    image::SampledImage,
    pipeline::{GraphicsPipeline, PipelineDescription},
};
use crate::{
    error::{Result, VulkanError},
    math::{Vec2, Vec4},
};
use ash::vk;

pub const MAX_TEXT_GLYPHS: usize = 2048;
const GLYPH_SIZE: usize = 8;
const LINE_ADVANCE: f32 = 10.0;
const ATLAS_COLUMNS: usize = 16;
const ATLAS_WIDTH: usize = ATLAS_COLUMNS * GLYPH_SIZE;
const ATLAS_HEIGHT: usize = 8 * GLYPH_SIZE;
const VERTICES_PER_GLYPH: usize = 6;
const FONT: &[u8; 1024] = include_bytes!("../../assets/font8x8.bin");
const ATLAS_BINDING: u32 = 0;

/// One frame's text; unsupported characters use `?`. Scale multiplies the 8-pixel font.
/// Newlines advance 10 pixels times scale. Color is linear RGBA, each channel in [0, 1].
pub struct TextDraw<'a> {
    pub text: &'a str,
    pub position: Vec2,
    pub scale: f32,
    pub color: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TextVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

fn atlas_pixels() -> Vec<u8> {
    let mut pixels = vec![0; ATLAS_WIDTH * ATLAS_HEIGHT];
    for glyph in 0..128 {
        for y in 0..GLYPH_SIZE {
            for x in 0..GLYPH_SIZE {
                let index = (glyph / ATLAS_COLUMNS * GLYPH_SIZE + y) * ATLAS_WIDTH
                    + glyph % ATLAS_COLUMNS * GLYPH_SIZE
                    + x;
                pixels[index] = if FONT[glyph * GLYPH_SIZE + y] & (1 << x) != 0 {
                    255
                } else {
                    0
                };
            }
        }
    }
    pixels
}

pub(super) fn layout_text(
    draws: &[TextDraw<'_>],
    extent: vk::Extent2D,
    vertices: &mut Vec<TextVertex>,
) -> Result<()> {
    vertices.clear();
    if extent.width == 0 || extent.height == 0 {
        return Ok(());
    }
    for draw in draws {
        let position = [draw.position[0], draw.position[1]];
        let color = [draw.color[0], draw.color[1], draw.color[2], draw.color[3]];
        if !draw.scale.is_finite()
            || draw.scale <= 0.0
            || !position.iter().all(|v| v.is_finite())
            || !color
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        {
            return Err(VulkanError::ValidationError(
                "text requires finite position, positive scale, and RGBA in [0,1]".into(),
            ));
        }
        let size = GLYPH_SIZE as f32 * draw.scale;
        if !size.is_finite() {
            return Err(VulkanError::ValidationError(
                "text scale is too large".into(),
            ));
        }
        let [mut x, mut y] = position;
        for ch in draw.text.chars() {
            if ch == '\n' {
                x = position[0];
                y += LINE_ADVANCE * draw.scale;
                continue;
            }
            let glyph = if ch.is_ascii_graphic() || ch == ' ' {
                ch as usize
            } else {
                b'?' as usize
            };
            if ch != ' '
                && x < extent.width as f32
                && y < extent.height as f32
                && x + size > 0.0
                && y + size > 0.0
            {
                if vertices.len() == MAX_TEXT_GLYPHS * VERTICES_PER_GLYPH {
                    return Err(VulkanError::ValidationError(format!(
                        "text exceeds {MAX_TEXT_GLYPHS} visible glyphs"
                    )));
                }
                let left = x / extent.width as f32 * 2.0 - 1.0;
                let top = y / extent.height as f32 * 2.0 - 1.0;
                let right = (x + size) / extent.width as f32 * 2.0 - 1.0;
                let bottom = (y + size) / extent.height as f32 * 2.0 - 1.0;
                let u = (glyph % ATLAS_COLUMNS * GLYPH_SIZE) as f32 / ATLAS_WIDTH as f32;
                let v = (glyph / ATLAS_COLUMNS * GLYPH_SIZE) as f32 / ATLAS_HEIGHT as f32;
                let du = GLYPH_SIZE as f32 / ATLAS_WIDTH as f32;
                let dv = GLYPH_SIZE as f32 / ATLAS_HEIGHT as f32;
                for (position, uv) in [
                    ([left, top], [u, v]),
                    ([right, top], [u + du, v]),
                    ([right, bottom], [u + du, v + dv]),
                    ([left, top], [u, v]),
                    ([right, bottom], [u + du, v + dv]),
                    ([left, bottom], [u, v + dv]),
                ] {
                    vertices.push(TextVertex {
                        position,
                        uv,
                        color,
                    });
                }
            }
            x += size;
        }
    }
    Ok(())
}

fn text_pipeline(
    device: &VulkanDevice,
    color: vk::Format,
    depth: vk::Format,
    layout: vk::DescriptorSetLayout,
) -> Result<GraphicsPipeline> {
    let attributes = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(TextVertex, position) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(TextVertex, uv) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(std::mem::offset_of!(TextVertex, color) as u32),
    ];
    GraphicsPipeline::create(
        device,
        color,
        depth,
        &[layout],
        PipelineDescription {
            vertex_shader: include_bytes!("../../shaders/text.vert.spv"),
            fragment_shader: include_bytes!("../../shaders/text.frag.spv"),
            binding: vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<TextVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
            attributes: &attributes,
            depth_test: false,
            cull_mode: vk::CullModeFlags::NONE,
            blend: true,
            push_ranges: &[],
        },
    )
}

pub(super) struct TextRenderer {
    image: Option<SampledImage>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    buffers: Vec<Buffer>,
    pipeline: Option<GraphicsPipeline>,
    formats: (vk::Format, vk::Format),
}

impl TextRenderer {
    pub fn new(
        device: &VulkanDevice,
        allocator: &mut Allocator,
        color: vk::Format,
        depth: vk::Format,
    ) -> Result<Self> {
        let dev = &device.device;
        let mut renderer = Self {
            image: None,
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            buffers: Vec::new(),
            pipeline: None,
            formats: (color, depth),
        };
        let result = (|| {
            renderer.image = Some(SampledImage::new_r8(
                device,
                allocator,
                vk::Extent2D {
                    width: ATLAS_WIDTH as u32,
                    height: ATLAS_HEIGHT as u32,
                },
                &atlas_pixels(),
            )?);
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(ATLAS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            renderer.layout = unsafe {
                dev.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)];
            renderer.pool = unsafe {
                dev.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
            }?;
            renderer.set = unsafe {
                dev.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(renderer.pool)
                        .set_layouts(&[renderer.layout]),
                )
            }?[0];
            let image = renderer.image.as_ref().unwrap();
            let images = [vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .sampler(image.sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            unsafe {
                dev.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(renderer.set)
                        .dst_binding(ATLAS_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&images)],
                    &[],
                )
            };
            for _ in 0..FRAMES_IN_FLIGHT {
                renderer.buffers.push(Buffer::new_host_visible_mapped(
                    device,
                    allocator,
                    (MAX_TEXT_GLYPHS * VERTICES_PER_GLYPH * std::mem::size_of::<TextVertex>())
                        as u64,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )?);
            }
            renderer.pipeline = Some(text_pipeline(device, color, depth, renderer.layout)?);
            Ok(())
        })();
        if let Err(error) = result {
            renderer.destroy(device, allocator);
            return Err(error);
        }
        Ok(renderer)
    }

    pub fn ensure_formats(
        &mut self,
        device: &VulkanDevice,
        color: vk::Format,
        depth: vk::Format,
    ) -> Result<()> {
        if self.formats != (color, depth) {
            let pipeline = text_pipeline(device, color, depth, self.layout)?;
            self.pipeline = Some(pipeline);
            self.formats = (color, depth);
        }
        Ok(())
    }

    /// Frame's fence must have signaled before writing this slot.
    pub fn update(&mut self, frame: usize, vertices: &[TextVertex]) -> Result<()> {
        self.buffers[frame].write_mapped(bytemuck::cast_slice(vertices))
    }

    /// Record after the scene in the same rendering scope; viewport/scissor are already set.
    pub unsafe fn record(
        &self,
        dev: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        vertex_count: u32,
    ) {
        if vertex_count == 0 {
            return;
        }
        let pipeline = self.pipeline.as_ref().unwrap();
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
        dev.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[self.set],
            &[],
        );
        dev.cmd_bind_vertex_buffers(cmd, 0, &[self.buffers[frame].buffer], &[0]);
        dev.cmd_draw(cmd, vertex_count, 1, 0, 0);
    }

    pub fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        // Drop the pipeline first (its layout must go before the pool/layout below).
        drop(self.pipeline);
        for buffer in self.buffers {
            buffer.destroy(device, allocator);
        }
        unsafe {
            device.device.destroy_descriptor_pool(self.pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
        if let Some(image) = self.image {
            image.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn draw(text: &str) -> TextDraw<'_> {
        TextDraw {
            text,
            position: Vec2::new(0.0, 0.0),
            scale: 1.0,
            color: Vec4::new(1.0, 0.5, 0.0, 1.0),
        }
    }
    fn layout(draws: &[TextDraw<'_>]) -> Result<Vec<TextVertex>> {
        let mut vertices = Vec::new();
        layout_text(
            draws,
            vk::Extent2D {
                width: 100,
                height: 100,
            },
            &mut vertices,
        )?;
        Ok(vertices)
    }
    #[test]
    fn text_vertex_and_shaders_match_inputs() {
        assert_eq!(std::mem::size_of::<TextVertex>(), 32);
        assert_eq!(std::mem::offset_of!(TextVertex, position), 0);
        assert_eq!(std::mem::offset_of!(TextVertex, uv), 8);
        assert_eq!(std::mem::offset_of!(TextVertex, color), 16);
        for bytes in [
            include_bytes!("../../shaders/text.vert.spv").as_slice(),
            include_bytes!("../../shaders/text.frag.spv").as_slice(),
        ] {
            let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
            assert_eq!(words[0], 0x0723_0203);
        }
    }

    #[test]
    fn new_frame_replaces_old_text_and_zero_extent_is_empty() {
        let mut vertices = layout(&[draw("A")]).unwrap();
        layout_text(
            &[],
            vk::Extent2D {
                width: 100,
                height: 100,
            },
            &mut vertices,
        )
        .unwrap();
        assert!(vertices.is_empty());
        layout_text(
            &[draw("A")],
            vk::Extent2D {
                width: 0,
                height: 100,
            },
            &mut vertices,
        )
        .unwrap();
        assert!(vertices.is_empty());
    }

    #[test]
    fn separate_draws_preserve_position_color_and_partial_clipping() {
        let mut second = draw("B");
        second.position = Vec2::new(-4.0, 20.0);
        second.color = Vec4::new(0.2, 0.3, 0.4, 0.5);
        let vertices = layout(&[draw("A"), second]).unwrap();
        assert_eq!(vertices.len(), 12);
        assert_eq!(vertices[6].color, [0.2, 0.3, 0.4, 0.5]);
        assert!((vertices[6].position[0] + 1.08).abs() < 0.00001);
        assert!((vertices[6].position[1] + 0.6).abs() < 0.00001);
    }

    #[test]
    fn atlas_preserves_glyph_bits() {
        let pixels = atlas_pixels();
        for (glyph, row, col) in [(b'A' as usize, 2, 3), (b'?' as usize, 4, 5)] {
            assert_eq!(
                pixels[(glyph / ATLAS_COLUMNS * GLYPH_SIZE + row) * ATLAS_WIDTH
                    + glyph % ATLAS_COLUMNS * GLYPH_SIZE
                    + col],
                if FONT[glyph * GLYPH_SIZE + row] & (1 << col) != 0 {
                    255
                } else {
                    0
                }
            );
        }
        assert!(pixels.contains(&255));
    }
    #[test]
    fn positions_newlines_color_and_fallback() {
        let vertices = layout(&[draw("A\n☃")]).unwrap();
        assert_eq!(vertices.len(), 12);
        assert_eq!(vertices[0].position, [-1.0, -1.0]);
        assert_eq!(vertices[6].position, [-1.0, -0.8]);
        assert_eq!(vertices[0].color, [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(vertices[6].uv, layout(&[draw("?")]).unwrap()[0].uv);
    }
    #[test]
    fn empty_clipped_and_scaled_text() {
        assert!(layout(&[]).unwrap().is_empty());
        assert!(layout(&[draw(" \n")]).unwrap().is_empty());
        let mut offscreen = draw("A");
        offscreen.position[0] = 100.0;
        assert!(layout(&[offscreen]).unwrap().is_empty());
        let mut scaled = draw("A");
        scaled.scale = 2.0;
        assert!((layout(&[scaled]).unwrap()[1].position[0] - -0.68).abs() < 0.00001);
    }
    #[test]
    fn rejects_bad_input_and_capacity_overflow() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut invalid = draw("A");
            invalid.scale = scale;
            assert!(layout(&[invalid]).is_err());
        }
        let mut invalid = draw("A");
        invalid.position[0] = f32::NAN;
        assert!(layout(&[invalid]).is_err());
        let mut invalid = draw("A");
        invalid.color[3] = 2.0;
        assert!(layout(&[invalid]).is_err());
        let draws: Vec<_> = (0..MAX_TEXT_GLYPHS).map(|_| draw("A")).collect();
        assert_eq!(
            layout(&draws).unwrap().len(),
            MAX_TEXT_GLYPHS * VERTICES_PER_GLYPH
        );
        let draws: Vec<_> = (0..=MAX_TEXT_GLYPHS).map(|_| draw("A")).collect();
        assert!(layout(&draws).is_err());
    }
}
