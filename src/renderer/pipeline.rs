// Graphics pipeline creation. See specs/001-graphics-pipeline.md for the full design rationale
// (dynamic rendering vs. render pass, precompiled SPIR-V vs. build-time shaderc, why there's no
// `extent` parameter).

use crate::error::Result; // TODO(spec 001): you'll also need VulkanError once `new()` below maps
                          // real ash::vk::Result errors -- add it back to this import then.
use crate::renderer::device::VulkanDevice;
use ash::vk;

// TODO(spec 001, step 0 -- do this first, it's a prerequisite for everything else here):
// compile the GLSL in `shaders/` to SPIR-V. `glslc` isn't installed in this environment; the
// pragmatic substitute is `glslang-tools` (apt package, ships `glslangValidator`):
//
//     sudo apt-get install -y glslang-tools
//     glslangValidator -V shaders/triangle.vert -o shaders/triangle.vert.spv
//     glslangValidator -V shaders/triangle.frag -o shaders/triangle.frag.spv
//
// (If you do have the real Vulkan SDK's `glslc` available instead, the equivalent is
// `glslc shaders/triangle.vert -o shaders/triangle.vert.spv`.) Commit the two `.spv` files --
// see specs/001-graphics-pipeline.md's "Decision: precompiled SPIR-V" for why they're checked in
// rather than compiled by a build.rs.
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
    pub fn new(_device: &VulkanDevice, _color_format: vk::Format) -> Result<Self> {
        // TODO(spec 001): load both shader modules. Use `ash::util::read_spv`, NOT
        // `bytemuck::cast_slice` -- `include_bytes!` gives `&[u8]` with no alignment guarantee,
        // and `VkShaderModuleCreateInfo::code` needs `&[u32]`. `read_spv` handles this correctly
        // regardless of input alignment; a raw cast would be UB if the bytes happen not to be
        // 4-byte aligned. Source: https://docs.rs/ash/latest/ash/util/fn.read_spv.html
        //
        //     let vertex_code = ash::util::read_spv(&mut std::io::Cursor::new(VERTEX_SHADER))
        //         .map_err(|e| VulkanError::InitializationError(format!("bad vertex SPIR-V: {e}")))?;
        //     let vertex_module = Self::create_shader_module(&device.device, &vertex_code)?;
        //     // ...same for FRAGMENT_SHADER...
        //
        // Both shader modules are only needed *during* pipeline creation (see the Vulkan objects
        // table in specs/001-graphics-pipeline.md) -- destroy them with `device.destroy_shader_module`
        // right after `create_graphics_pipelines` below returns, before this function returns.

        // TODO(spec 001): build the two `VkPipelineShaderStageCreateInfo` entries (VERTEX stage
        // using vertex_module, FRAGMENT stage using fragment_module), entry point `c"main"`.

        // TODO(spec 001): vertex input state is EMPTY -- no bindings, no attributes.
        // `vk::PipelineVertexInputStateCreateInfo::default()` with nothing else set is correct;
        // the shader hardcodes positions via `gl_VertexIndex` (see shaders/triangle.vert).

        // TODO(spec 001): input assembly = `TRIANGLE_LIST`, `primitive_restart_enable(false)`.

        // TODO(spec 001): viewport state needs `viewport_count(1)` and `scissor_count(1)` but NO
        // actual `VkViewport`/`VkRect2D` values -- those come from dynamic state instead (see the
        // "No `extent` parameter" note on this function's doc comment in the spec). Pair this
        // with a `VkPipelineDynamicStateCreateInfo` listing `vk::DynamicState::VIEWPORT` and
        // `vk::DynamicState::SCISSOR`. Source for why dynamic viewport/scissor is the standard
        // modern approach (not just a shortcut): Vulkan Tutorial's "Fixed functions" chapter,
        // dynamic state section, https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions

        // TODO(spec 001): rasterization state -- `polygon_mode(FILL)`, `cull_mode(BACK)`,
        // `front_face(CLOCKWISE)` (matches the hardcoded triangle's winding order in
        // shaders/triangle.vert -- if the triangle doesn't render, a mismatched front_face
        // against cull_mode is a likely first suspect), `line_width(1.0)`.

        // TODO(spec 001): multisample state -- disabled for now (`rasterization_samples(TYPE_1)`,
        // `sample_shading_enable(false)`). MSAA is a Phase 4 roadmap item, not today.

        // TODO(spec 001): color blend state -- one `VkPipelineColorBlendAttachmentState` with
        // `blend_enable(false)` and `color_write_mask` set to all four channels (R|G|B|A).

        // TODO(spec 001): pipeline layout -- empty. No descriptor set layouts, no push constant
        // ranges (Phase 2 adds these once there's uniform data to pass in).
        //
        //     let pipeline_layout = unsafe {
        //         device.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        //             .map_err(VulkanError::from)?
        //     };

        // TODO(spec 001): the dynamic-rendering-specific part -- chain
        // `vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&[color_format])`
        // into `VkGraphicsPipelineCreateInfo` via `.push_next(...)`, and do NOT call
        // `.render_pass(...)` anywhere (that's the classic render-pass API this spec deliberately
        // isn't using). Source: same Vulkan-Samples dynamic_rendering link as device.rs's TODO.

        // TODO(spec 001): `create_graphics_pipelines` takes a `vk::PipelineCache` (pass
        // `vk::PipelineCache::null()` -- pipeline caching is a later optimization, not today) and
        // a slice of create-infos (just the one, for now). Destroy both shader modules
        // immediately after this call succeeds.

        let _ = (VERTEX_SHADER, FRAGMENT_SHADER);
        todo!(
            "implement per specs/001-graphics-pipeline.md -- see the TODO comments above, in order"
        )
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        // Pipeline before layout: a layout can't be destroyed while a pipeline created from it
        // still exists -- same "children before parents" rule as the mod.rs VulkanRenderer fix,
        // just one level shallower here. This part isn't a TODO -- it's already correct and will
        // stay correct once `new()` above actually populates real handles.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
