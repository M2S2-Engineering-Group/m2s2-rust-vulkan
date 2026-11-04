// Graphics pipeline creation. See specs/001-graphics-pipeline.md for the full design rationale
// (dynamic rendering vs. render pass, precompiled SPIR-V vs. build-time shaderc, why there's no
// `extent` parameter).

use crate::error::{Result, VulkanError};
use crate::renderer::buffer::Vertex;
use crate::renderer::device::VulkanDevice;
use ash::vk;
use std::io::Cursor;

// Precompiled SPIR-V, committed to the repo (see specs/001 "Decision: precompiled SPIR-V").
// Recompile after editing: glslc shaders/triangle.vert -o shaders/triangle.vert.spv
const VERTEX_SHADER: &[u8] = include_bytes!("../../shaders/triangle.vert.spv");
const FRAGMENT_SHADER: &[u8] = include_bytes!("../../shaders/triangle.frag.spv");

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
    /// MVP uniform layout from `uniform.rs`, referenced by the pipeline layout so the shader can read
    /// `set = 0`. No `extent` parameter: viewport and scissor are dynamic state, set per-frame in
    /// `command.rs` (spec 002).
    pub fn new(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        let device = &device.device;

        // include_bytes! gives &[u8] with no alignment guarantee; VkShaderModuleCreateInfo wants
        // &[u32]. read_spv reads the little-endian SPIR-V words correctly regardless of alignment
        // (a raw bytemuck cast would be UB on an unaligned slice).
        let vertex_code = ash::util::read_spv(&mut Cursor::new(VERTEX_SHADER))
            .map_err(|e| VulkanError::InitializationError(format!("bad vertex SPIR-V: {e}")))?;
        let fragment_code = ash::util::read_spv(&mut Cursor::new(FRAGMENT_SHADER))
            .map_err(|e| VulkanError::InitializationError(format!("bad fragment SPIR-V: {e}")))?;

        let vertex_module = Self::create_shader_module(device, &vertex_code)?;
        let fragment_module = Self::create_shader_module(device, &fragment_code)?;

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

        // Vertex input: one per-vertex binding + the position/color attributes, from the Vertex
        // type (buffer.rs). Must stay alive until create_graphics_pipelines reads through it below.
        let binding_descriptions = [Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

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
            .cull_mode(vk::CullModeFlags::BACK)
            // COUNTER_CLOCKWISE: the perspective Y-flip (see math::perspective) inverts screen-space
            // winding, so standard CCW-outward geometry is front-facing under this setting. (Spec 006
            // changed this from CLOCKWISE, which suited the pre-projection clip-space quad.)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        // Depth test enabled: fragments nearer the camera (smaller depth, LESS) win and write their
        // depth back. Required for the cube's far faces to be occluded by its near ones (spec 006).
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        // One descriptor set (set = 0): the per-frame MVP uniform block (spec 006).
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
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

        // Shader modules are only needed during pipeline creation — destroy them now, regardless of
        // whether creation succeeded.
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }

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
