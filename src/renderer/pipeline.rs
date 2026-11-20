// Graphics pipeline creation. See specs/001-graphics-pipeline.md for the full design rationale
// (dynamic rendering vs. render pass, precompiled SPIR-V vs. build-time shaderc, why there's no
// `extent` parameter).

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use crate::math::Vec3;
use crate::math::Vec4;
use crate::renderer::buffer::Vertex;
use crate::renderer::device::VulkanDevice;
use ash::vk;
use std::io::Cursor;

// Precompiled SPIR-V, committed to the repo (see specs/001 "Decision: precompiled SPIR-V").
// Recompile after editing: glslc shaders/triangle.vert -o shaders/triangle.vert.spv
// These two are the SHADOWS_ENABLED-undefined variant; see CLAUDE.md's shader recompile note for
// the second, -DSHADOWS_ENABLED-compiled pair (spec 017).
const VERTEX_SHADER: &[u8] = include_bytes!("../../shaders/triangle.vert.spv");
const FRAGMENT_SHADER: &[u8] = include_bytes!("../../shaders/triangle.frag.spv");

// Shadow-pass shaders (spec 017). SHADOW_VERTEX_SHADER/SHADOW_FRAGMENT_SHADER are the main scene
// pipeline's SHADOWS_ENABLED-compiled variant (samples the shadow map via set 3);
// SHADOW_DEPTH_VERTEX_SHADER is the depth-only shadow pass itself (no fragment shader).
const SHADOW_VERTEX_SHADER: &[u8] = include_bytes!("../../shaders/triangle_shadow.vert.spv");
const SHADOW_FRAGMENT_SHADER: &[u8] = include_bytes!("../../shaders/triangle_shadow.frag.spv");
const SHADOW_DEPTH_VERTEX_SHADER: &[u8] = include_bytes!("../../shaders/shadow_depth.vert.spv");

/// Per-instance GPU payload for instanced draws (plan 016): one row-major `Mat4` model transform,
/// unpacked into four separate `vec4`s (`model_row0..3`) because the GLSL side reads four distinct
/// `vec4` instance-rate attribute locations, not one `mat4` attribute, plus the specular
/// color/shininess packed into a fifth `vec4` (`xyz` = color, `w` = shininess). 80 bytes, offsets
/// 0/16/32/48/64.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
    pub model_row0: Vec4,
    pub model_row1: Vec4,
    pub model_row2: Vec4,
    pub model_row3: Vec4,
    pub specular: Vec4,
}

impl From<(Mat4, Vec3, f32)> for InstanceData {
    /// Extracts each row of a row-major `Mat4` as a `Vec4` via `m2s2-math`'s `Index<usize>` impl on
    /// `Matrix4x4`, which returns row `i`'s four components as `&[f32]` (row-major storage, same
    /// convention as `CameraUniforms`'s `row_major` GLSL qualifier) — `Vec4::from_slice` copies that
    /// into an owned `Vec4`. `specular_color` and `shininess` pack into one `vec4` (`xyz`/`w`).
    fn from((model, specular_color, shininess): (Mat4, Vec3, f32)) -> Self {
        Self {
            model_row0: Vec4::from_slice(&model[0]),
            model_row1: Vec4::from_slice(&model[1]),
            model_row2: Vec4::from_slice(&model[2]),
            model_row3: Vec4::from_slice(&model[3]),
            specular: Vec4::new(
                specular_color[0],
                specular_color[1],
                specular_color[2],
                shininess,
            ),
        }
    }
}

/// Instance binding's five attribute descriptions (locations 4-8), offsets taken from
/// `InstanceData`'s own field offsets via `std::mem::offset_of!` so they can't drift from the
/// struct's field order.
fn instance_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 5] {
    let format = vk::Format::R32G32B32A32_SFLOAT;
    [
        vk::VertexInputAttributeDescription::default()
            .location(4)
            .binding(1)
            .format(format)
            .offset(std::mem::offset_of!(InstanceData, model_row0) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(5)
            .binding(1)
            .format(format)
            .offset(std::mem::offset_of!(InstanceData, model_row1) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(6)
            .binding(1)
            .format(format)
            .offset(std::mem::offset_of!(InstanceData, model_row2) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(7)
            .binding(1)
            .format(format)
            .offset(std::mem::offset_of!(InstanceData, model_row3) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(8)
            .binding(1)
            .format(format)
            .offset(std::mem::offset_of!(InstanceData, specular) as u32),
    ]
}

/// One instance binding, consumed one `InstanceData` per instance (not per vertex).
fn instance_binding_description() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription::default()
        .binding(1)
        .stride(std::mem::size_of::<InstanceData>() as u32)
        .input_rate(vk::VertexInputRate::INSTANCE)
}

/// The scene vertex-input layout (per-vertex `Vertex` binding 0 + per-instance `InstanceData`
/// binding 1), shared by every scene-geometry pipeline variant -- the normal and shadow-enabled
/// main pipelines and the depth-only shadow pipeline all draw the same geometry through the same
/// instance buffer, so this is the one place that layout is assembled.
fn scene_bindings_and_attributes() -> (
    [vk::VertexInputBindingDescription; 2],
    Vec<vk::VertexInputAttributeDescription>,
) {
    let bindings = [
        Vertex::binding_description(),
        instance_binding_description(),
    ];
    let attributes = Vertex::attribute_descriptions()
        .into_iter()
        .chain(instance_attribute_descriptions())
        .collect();
    (bindings, attributes)
}

pub(super) struct PipelineDescription<'a> {
    pub vertex_shader: &'a [u8],
    pub fragment_shader: Option<&'a [u8]>,
    pub bindings: &'a [vk::VertexInputBindingDescription],
    pub attributes: &'a [vk::VertexInputAttributeDescription],
    pub depth_test: bool,
    pub cull_mode: vk::CullModeFlags,
    pub blend: bool,
    pub push_ranges: &'a [vk::PushConstantRange],
}

#[allow(dead_code)] // Consumed by Task 4 (depth-only pipeline)
pub(super) fn shader_stage_count(fragment_shader: Option<&[u8]>) -> usize {
    1 + fragment_shader.is_some() as usize
}

pub struct GraphicsPipeline {
    // pub(crate): command.rs (spec 002) needs to read these to record vkCmdBindPipeline. Not
    // `pub` -- nothing outside this crate should reach into a Vulkan handle directly.
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    device: ash::Device, // cloned handle, kept only so Drop can destroy the two fields above
}

impl GraphicsPipeline {
    /// `color_formats` should contain the swapchain's format (`VulkanSwapchain::format`); `depth_format`
    /// the depth attachment's (`DepthResources::format`). `set_layouts` are referenced by the
    /// pipeline layout in order, so the shader can read `set = 0`, `set = 1`, etc. (the per-frame
    /// camera uniform layout from `uniform.rs`, then the per-object texture layout from
    /// `texture.rs`, then the per-frame light layout from `light.rs`). No `extent` parameter:
    /// viewport and scissor are dynamic state, set per-frame
    /// in `command.rs` (spec 002).
    pub fn new(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self> {
        let (bindings, attributes) = scene_bindings_and_attributes();
        Self::create(
            device,
            &[color_format],
            depth_format,
            set_layouts,
            PipelineDescription {
                vertex_shader: VERTEX_SHADER,
                fragment_shader: Some(FRAGMENT_SHADER),
                bindings: &bindings,
                attributes: &attributes,
                depth_test: true,
                cull_mode: vk::CullModeFlags::BACK,
                blend: false,
                push_ranges: &[],
            },
        )
    }

    /// The shadow-enabled main-pipeline variant: same vertex-input layout and set 0/1/2 (camera,
    /// texture, light) as `new`, plus an additional set (the shadow sampler + light-VP UBO --
    /// `ShadowResources::descriptor_set_layout`) and shaders compiled with `SHADOWS_ENABLED`
    /// defined (`triangle_shadow.{vert,frag}.spv`). `VulkanRenderer` selects this over `new`
    /// whenever a directional shadow is currently configured.
    pub(super) fn new_with_shadow(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self> {
        let (bindings, attributes) = scene_bindings_and_attributes();
        Self::create(
            device,
            &[color_format],
            depth_format,
            set_layouts,
            PipelineDescription {
                vertex_shader: SHADOW_VERTEX_SHADER,
                fragment_shader: Some(SHADOW_FRAGMENT_SHADER),
                bindings: &bindings,
                attributes: &attributes,
                depth_test: true,
                cull_mode: vk::CullModeFlags::BACK,
                blend: false,
                push_ranges: &[],
            },
        )
    }

    /// The depth-only shadow pass's own pipeline: draws the same scene geometry (same
    /// vertex-input layout as `new`/`new_with_shadow`, through the same instance buffer) into the
    /// shadow map, with no fragment shader and no color attachments (`color_formats: &[]` -- see
    /// `create`'s dynamic-rendering-attachment-count handling below). Back-face culling, per spec
    /// 017's initial bias-direction choice.
    pub(super) fn new_shadow_depth(
        device: &VulkanDevice,
        depth_format: vk::Format,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self> {
        let (bindings, attributes) = scene_bindings_and_attributes();
        Self::create(
            device,
            &[],
            depth_format,
            set_layouts,
            PipelineDescription {
                vertex_shader: SHADOW_DEPTH_VERTEX_SHADER,
                fragment_shader: None,
                bindings: &bindings,
                attributes: &attributes,
                depth_test: true,
                cull_mode: vk::CullModeFlags::BACK,
                blend: false,
                push_ranges: &[],
            },
        )
    }

    pub(super) fn create(
        device: &VulkanDevice,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        set_layouts: &[vk::DescriptorSetLayout],
        description: PipelineDescription<'_>,
    ) -> Result<Self> {
        let device = &device.device;

        // include_bytes! gives &[u8] with no alignment guarantee; VkShaderModuleCreateInfo wants
        // &[u32]. read_spv reads the little-endian SPIR-V words correctly regardless of alignment
        // (a raw bytemuck cast would be UB on an unaligned slice).
        let vertex_code = ash::util::read_spv(&mut Cursor::new(description.vertex_shader))
            .map_err(|e| VulkanError::InitializationError(format!("bad vertex SPIR-V: {e}")))?;

        let vertex_module = Self::create_shader_module(device, &vertex_code)?;
        let _vertex_cleanup = super::cleanup::Cleanup::new(|| unsafe {
            device.destroy_shader_module(vertex_module, None)
        });

        let entry_name = c"main";
        let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
            vec![vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(entry_name)];

        let fragment_module;
        let _fragment_cleanup = if let Some(fragment_shader) = description.fragment_shader {
            let fragment_code =
                ash::util::read_spv(&mut Cursor::new(fragment_shader)).map_err(|e| {
                    VulkanError::InitializationError(format!("bad fragment SPIR-V: {e}"))
                })?;
            fragment_module = Self::create_shader_module(device, &fragment_code)?;
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_module)
                    .name(entry_name),
            );
            Some(super::cleanup::Cleanup::new(|| unsafe {
                device.destroy_shader_module(fragment_module, None)
            }))
        } else {
            None
        };

        // Shared construction accepts scene or text vertex layouts; the caller keeps descriptions
        // alive until create_graphics_pipelines reads them.
        let binding_descriptions = description.bindings;
        let attribute_descriptions = description.attributes;
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport/scissor are dynamic: declare the counts here, supply the actual rects per-frame
        // via vkCmdSetViewport/vkCmdSetScissor (spec 002). Avoids rebuilding on window resize.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(description.cull_mode)
            // COUNTER_CLOCKWISE: the perspective Y-flip (see math::perspective) inverts screen-space
            // winding, so standard CCW-outward geometry is front-facing under this setting. (Spec 006
            // changed this from CLOCKWISE, which suited the pre-projection clip-space quad.)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        // Scene geometry tests/writes depth; text overlays disable both.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(description.depth_test)
            .depth_write_enable(description.depth_test)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        // Vulkan requires colorBlendAttachmentCount to equal the dynamic-rendering
        // colorAttachmentCount (color_formats.len()) exactly -- the depth-only shadow pipeline
        // passes an empty color_formats, so it needs zero blend-attachment entries here, not one.
        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> =
            if color_formats.is_empty() {
                Vec::new()
            } else {
                vec![vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(description.blend)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)]
            };
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        // Set 0 contains the camera UBO for scene geometry or the sampled font atlas for text;
        // scene geometry also has a set 1 per-object texture.
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(description.push_ranges);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(VulkanError::from)?
        };

        // Dynamic rendering: the pipeline declares the color + depth attachment *formats* it targets
        // via push_next, instead of being bound to a VkRenderPass + subpass. No .render_pass(...).
        // An empty color_formats slice produces colorAttachmentCount: 0, which is the correct
        // "depth-only, no color attachments" Vulkan configuration.
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(color_formats)
            .depth_attachment_format(depth_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let pipeline_result = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };

        // create_graphics_pipelines reports failure as (partial pipelines, result) — clean up any
        // partial pipeline plus the layout so neither leaks before returning the error.
        let pipeline = match pipeline_result {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, result)) => {
                for p in pipelines {
                    if p != vk::Pipeline::null() {
                        unsafe { device.destroy_pipeline(p, None) };
                    }
                }
                unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
                return Err(VulkanError::from(result));
            }
        };

        log::debug!("Graphics pipeline created");

        Ok(Self {
            pipeline,
            pipeline_layout,
            device: device.clone(),
        })
    }

    fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe {
            device
                .create_shader_module(&create_info, None)
                .map_err(VulkanError::from)
        }
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        // Pipeline before layout: a layout can't be destroyed while a pipeline created from it
        // still exists -- same "children before parents" rule as the mod.rs VulkanRenderer fix,
        // just one level shallower here.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_stage_count_is_one_without_fragment_two_with() {
        assert_eq!(shader_stage_count(None), 1);
        assert_eq!(shader_stage_count(Some(&[0u8; 4])), 2);
    }

    #[test]
    fn instance_data_matches_shader_layout() {
        assert_eq!(std::mem::size_of::<InstanceData>(), 80);
        assert_eq!(std::mem::offset_of!(InstanceData, model_row0), 0);
        assert_eq!(std::mem::offset_of!(InstanceData, model_row1), 16);
        assert_eq!(std::mem::offset_of!(InstanceData, model_row2), 32);
        assert_eq!(std::mem::offset_of!(InstanceData, model_row3), 48);
        assert_eq!(std::mem::offset_of!(InstanceData, specular), 64);
    }

    #[test]
    fn instance_data_converts_rows_and_packs_specular() {
        let model = crate::math::translate(&Vec3::new(1.0, 2.0, 3.0));
        let specular_color = Vec3::new(0.4, 0.5, 0.6);
        let shininess = 32.0;
        let instance = InstanceData::from((model, specular_color, shininess));
        assert_eq!(instance.model_row0, Vec4::from_slice(&model[0]));
        assert_eq!(instance.model_row1, Vec4::from_slice(&model[1]));
        assert_eq!(instance.model_row2, Vec4::from_slice(&model[2]));
        assert_eq!(instance.model_row3, Vec4::from_slice(&model[3]));
        assert_eq!(
            instance.specular,
            Vec4::new(
                specular_color[0],
                specular_color[1],
                specular_color[2],
                shininess
            )
        );
    }

    #[test]
    fn instance_binding_is_binding_one_instance_rate() {
        let b = instance_binding_description();
        assert_eq!(b.binding, 1);
        assert_eq!(b.stride, std::mem::size_of::<InstanceData>() as u32);
        assert_eq!(b.input_rate, vk::VertexInputRate::INSTANCE);
    }

    #[test]
    fn instance_attributes_occupy_locations_4_through_8_on_binding_1() {
        let attrs = instance_attribute_descriptions();
        assert_eq!(attrs.len(), 5);
        for (i, attr) in attrs.iter().enumerate() {
            assert_eq!(attr.location, 4 + i as u32);
            assert_eq!(attr.binding, 1);
            assert_eq!(attr.format, vk::Format::R32G32B32A32_SFLOAT);
        }
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[1].offset, 16);
        assert_eq!(attrs[2].offset, 32);
        assert_eq!(attrs[3].offset, 48);
        assert_eq!(attrs[4].offset, 64);
    }

    // SPIR-V's first word is a fixed magic number; a correctly compiled .spv always starts with it.
    const SPIRV_MAGIC: u32 = 0x0723_0203;

    #[test]
    fn vertex_spirv_parses_and_starts_with_magic() {
        let words = ash::util::read_spv(&mut Cursor::new(VERTEX_SHADER))
            .expect("vertex .spv should parse via read_spv");
        assert_eq!(words.first().copied(), Some(SPIRV_MAGIC));
    }

    #[test]
    fn fragment_spirv_parses_and_starts_with_magic() {
        let words = ash::util::read_spv(&mut Cursor::new(FRAGMENT_SHADER))
            .expect("fragment .spv should parse via read_spv");
        assert_eq!(words.first().copied(), Some(SPIRV_MAGIC));
    }

    #[test]
    fn shadow_depth_vertex_spirv_parses_and_starts_with_magic() {
        let bytes = include_bytes!("../../shaders/shadow_depth.vert.spv");
        let words = ash::util::read_spv(&mut Cursor::new(&bytes[..]))
            .expect("shadow_depth.vert .spv should parse via read_spv");
        assert_eq!(words.first().copied(), Some(SPIRV_MAGIC));
    }

    #[test]
    fn triangle_shadow_vertex_and_fragment_spirv_parse_and_start_with_magic() {
        let v = include_bytes!("../../shaders/triangle_shadow.vert.spv");
        let f = include_bytes!("../../shaders/triangle_shadow.frag.spv");
        assert_eq!(
            ash::util::read_spv(&mut Cursor::new(&v[..]))
                .expect("triangle_shadow.vert .spv should parse via read_spv")
                .first()
                .copied(),
            Some(SPIRV_MAGIC)
        );
        assert_eq!(
            ash::util::read_spv(&mut Cursor::new(&f[..]))
                .expect("triangle_shadow.frag .spv should parse via read_spv")
                .first()
                .copied(),
            Some(SPIRV_MAGIC)
        );
    }
}
