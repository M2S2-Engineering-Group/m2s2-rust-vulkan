// Graphics pipeline creation. See specs/001-graphics-pipeline.md for the full design rationale
// (dynamic rendering vs. render pass, precompiled SPIR-V vs. build-time shaderc, why there's no
// `extent` parameter).

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use crate::renderer::buffer::Vertex;
use crate::renderer::device::VulkanDevice;
use ash::vk;
use std::io::Cursor;

// Precompiled SPIR-V, committed to the repo (see specs/001 "Decision: precompiled SPIR-V").
// Recompile after editing: glslc shaders/triangle.vert -o shaders/triangle.vert.spv
const VERTEX_SHADER: &[u8] = include_bytes!("../../shaders/triangle.vert.spv");
const FRAGMENT_SHADER: &[u8] = include_bytes!("../../shaders/triangle.frag.spv");

/// Per-draw model transform, copied into command state before each indexed draw.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ModelPushConstants {
    pub model: Mat4,
}

pub(crate) const MODEL_PUSH_OFFSET: u32 = 0;
pub(crate) const MODEL_PUSH_SIZE: u32 = std::mem::size_of::<ModelPushConstants>() as u32;
pub(crate) const MODEL_PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::VERTEX;

fn model_push_range() -> vk::PushConstantRange {
    vk::PushConstantRange::default()
        .offset(MODEL_PUSH_OFFSET)
        .size(MODEL_PUSH_SIZE)
        .stage_flags(MODEL_PUSH_STAGES)
}

fn validate_push_limit(limit: u32) -> Result<()> {
    if MODEL_PUSH_OFFSET + MODEL_PUSH_SIZE > limit {
        return Err(VulkanError::InitializationError(format!(
            "model push constants require {} bytes, device supports {limit}",
            MODEL_PUSH_OFFSET + MODEL_PUSH_SIZE,
        )));
    }
    Ok(())
}

pub(super) struct PipelineDescription<'a> {
    pub vertex_shader: &'a [u8],
    pub fragment_shader: &'a [u8],
    pub binding: vk::VertexInputBindingDescription,
    pub attributes: &'a [vk::VertexInputAttributeDescription],
    pub depth_test: bool,
    pub cull_mode: vk::CullModeFlags,
    pub blend: bool,
    pub push_ranges: &'a [vk::PushConstantRange],
}

pub struct GraphicsPipeline {
    // pub(crate): command.rs (spec 002) needs to read these to record vkCmdBindPipeline. Not
    // `pub` -- nothing outside this crate should reach into a Vulkan handle directly.
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    device: ash::Device, // cloned handle, kept only so Drop can destroy the two fields above
}

impl GraphicsPipeline {
    /// `color_format` should be the swapchain's format (`VulkanSwapchain::format`); `depth_format`
    /// the depth attachment's (`DepthResources::format`). `descriptor_set_layout` is the per-frame
    /// camera uniform layout from `uniform.rs`, referenced by the pipeline layout so the shader can read
    /// `set = 0`. No `extent` parameter: viewport and scissor are dynamic state, set per-frame in
    /// `command.rs` (spec 002).
    pub fn new(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        validate_push_limit(device.max_push_constants_size)?;
        Self::create(
            device,
            color_format,
            depth_format,
            descriptor_set_layout,
            PipelineDescription {
                vertex_shader: VERTEX_SHADER,
                fragment_shader: FRAGMENT_SHADER,
                binding: Vertex::binding_description(),
                attributes: &Vertex::attribute_descriptions(),
                depth_test: true,
                cull_mode: vk::CullModeFlags::BACK,
                blend: false,
                push_ranges: &[model_push_range()],
            },
        )
    }

    pub(super) fn create(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
        descriptor_set_layout: vk::DescriptorSetLayout,
        description: PipelineDescription<'_>,
    ) -> Result<Self> {
        let device = &device.device;

        // include_bytes! gives &[u8] with no alignment guarantee; VkShaderModuleCreateInfo wants
        // &[u32]. read_spv reads the little-endian SPIR-V words correctly regardless of alignment
        // (a raw bytemuck cast would be UB on an unaligned slice).
        let vertex_code = ash::util::read_spv(&mut Cursor::new(description.vertex_shader))
            .map_err(|e| VulkanError::InitializationError(format!("bad vertex SPIR-V: {e}")))?;
        let fragment_code = ash::util::read_spv(&mut Cursor::new(description.fragment_shader))
            .map_err(|e| VulkanError::InitializationError(format!("bad fragment SPIR-V: {e}")))?;

        let vertex_module = Self::create_shader_module(device, &vertex_code)?;
        let _vertex_cleanup = super::cleanup::Cleanup::new(|| unsafe {
            device.destroy_shader_module(vertex_module, None)
        });
        let fragment_module = Self::create_shader_module(device, &fragment_code)?;

        let _fragment_cleanup = super::cleanup::Cleanup::new(|| unsafe {
            device.destroy_shader_module(fragment_module, None)
        });
        let entry_name = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(entry_name),
        ];

        // Shared construction accepts scene or text vertex layouts; the caller keeps descriptions
        // alive until create_graphics_pipelines reads them.
        let binding_descriptions = [description.binding];
        let attribute_descriptions = description.attributes;
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
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

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(description.blend)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        // Set 0 contains the camera UBO for scene geometry or the sampled font atlas for text.
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(description.push_ranges);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(VulkanError::from)?
        };

        // Dynamic rendering: the pipeline declares the color + depth attachment *formats* it targets
        // via push_next, instead of being bound to a VkRenderPass + subpass. No .render_pass(...).
        let color_formats = [color_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
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
    fn model_payload_and_range_match_shader_layout() {
        assert_eq!(std::mem::size_of::<ModelPushConstants>(), 64);
        assert_eq!(std::mem::offset_of!(ModelPushConstants, model), 0);
        let range = model_push_range();
        assert_eq!(range.offset, 0);
        assert_eq!(range.size, 64);
        assert_eq!(range.stage_flags, vk::ShaderStageFlags::VERTEX);
    }

    #[test]
    fn push_limit_accepts_equal_or_larger_and_rejects_smaller() {
        assert!(validate_push_limit(MODEL_PUSH_SIZE).is_ok());
        assert!(validate_push_limit(MODEL_PUSH_SIZE + 4).is_ok());
        assert!(validate_push_limit(MODEL_PUSH_SIZE - 4).is_err());
        assert!(validate_push_limit(0).is_err());
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
}
