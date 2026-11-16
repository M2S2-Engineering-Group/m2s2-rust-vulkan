// Command pool/buffer management, synchronization, and the render loop. See
// specs/002-command-buffers-and-render-loop.md for the full design -- especially the "dynamic
// rendering gap" section, since this is the one part of the render loop that can't be a
// straight port of vulkan-tutorial.com's render-pass-based version.

use super::light::{DirectionalLight, Light, PointLight};
use super::mesh::{self, StoredMesh};
use super::scene::Object;
use super::text::{TextRenderer, TextVertex};
use super::texture::resolve_texture_set;
use crate::error::{Result, VulkanError};
use crate::renderer::depth::DepthResources;
use crate::renderer::device::VulkanDevice;
use crate::renderer::instance_buffer::InstanceBuffer;
use crate::renderer::pipeline::{GraphicsPipeline, InstanceData};
use crate::renderer::swapchain::VulkanSwapchain;
use crate::renderer::uniform::{CameraUniforms, Uniforms};
use ash::vk;

pub const FRAMES_IN_FLIGHT: usize = 2;

/// Result of one `draw_frame`: whether it rendered normally or the swapchain went out of date (the
/// window resized) and the caller should recreate it.
pub enum FrameOutcome {
    Rendered,
    SwapchainOutOfDate { submitted: bool },
}

impl FrameOutcome {
    pub(super) fn submitted(&self) -> bool {
        matches!(
            self,
            Self::Rendered | Self::SwapchainOutOfDate { submitted: true }
        )
    }
}

pub struct FrameContext {
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    image_available_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    // One render-finished semaphore PER SWAPCHAIN IMAGE (not per frame-in-flight): present waits on
    // it with no fence to report when it's consumed, so a per-frame semaphore could be reused while
    // a prior present still references it (validation flags this). Indexed by acquired image index;
    // recreated when the image count changes on swapchain recreate.
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    current_frame: usize,
    device: ash::Device, // cloned handle, kept only so Drop can destroy the fields above
}

impl FrameContext {
    pub fn new(device: &VulkanDevice, image_count: usize) -> Result<Self> {
        let dev = &device.device;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.graphics_queue_family)
            // RESET_COMMAND_BUFFER lets draw_frame reset one buffer per frame instead of the pool.
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            dev.create_command_pool(&pool_info, None)
                .map_err(VulkanError::from)?
        };

        let mut context = Self {
            command_pool,
            command_buffers: [vk::CommandBuffer::null(); FRAMES_IN_FLIGHT],
            image_available_semaphores: [vk::Semaphore::null(); FRAMES_IN_FLIGHT],
            render_finished_semaphores: Vec::new(),
            in_flight_fences: [vk::Fence::null(); FRAMES_IN_FLIGHT],
            current_frame: 0,
            device: dev.clone(),
        };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        context.command_buffers = unsafe {
            dev.allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::from)?
        }
        .try_into()
        .expect("allocate_command_buffers returns FRAMES_IN_FLIGHT buffers");

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // Created SIGNALED so the very first wait_for_fences (before any submit) returns
        // immediately instead of deadlocking.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for i in 0..FRAMES_IN_FLIGHT {
            unsafe {
                context.image_available_semaphores[i] = dev
                    .create_semaphore(&semaphore_info, None)
                    .map_err(VulkanError::from)?;
                context.in_flight_fences[i] = dev
                    .create_fence(&fence_info, None)
                    .map_err(VulkanError::from)?;
            }
        }
        // render-finished is per swapchain image (see the field comment).
        context.render_finished_semaphores = Self::create_present_semaphores(dev, image_count)?;

        Ok(context)
    }

    fn create_present_semaphores(dev: &ash::Device, count: usize) -> Result<Vec<vk::Semaphore>> {
        let info = vk::SemaphoreCreateInfo::default();
        super::cleanup::collect_resources(
            0..count,
            |_| unsafe { dev.create_semaphore(&info, None).map_err(VulkanError::from) },
            |semaphore| unsafe { dev.destroy_semaphore(semaphore, None) },
        )
    }

    /// Recreate the per-swapchain-image render-finished semaphores — the image count can change on a
    /// swapchain recreate. The caller must have waited for the device to be idle.
    pub(crate) fn recreate_present_semaphores(
        &mut self,
        device: &VulkanDevice,
        image_count: usize,
    ) -> Result<()> {
        let replacement = Self::create_present_semaphores(&device.device, image_count)?;
        for semaphore in std::mem::replace(&mut self.render_finished_semaphores, replacement) {
            unsafe { device.device.destroy_semaphore(semaphore, None) };
        }
        Ok(())
    }

    /// Runs one full frame against `swapchain`, using `pipeline` to draw. See
    /// specs/002's "Per-frame protocol" and "Recording sequence" sections.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_frame(
        &mut self,
        device: &VulkanDevice,
        pipeline: &GraphicsPipeline,
        swapchain: &VulkanSwapchain,
        depth: &DepthResources,
        uniforms: &mut Uniforms,
        camera: &CameraUniforms,
        light: &mut Light,
        directional_light: &DirectionalLight,
        point_lights: &[PointLight],
        instance_buffer: &mut InstanceBuffer,
        objects: &[Object],
        owner: u64,
        meshes: &[StoredMesh],
        texture_sets: &[vk::DescriptorSet],
        white_set: vk::DescriptorSet,
        mut text: Option<&mut TextRenderer>,
        text_vertices: &[TextVertex],
    ) -> Result<FrameOutcome> {
        let dev = &device.device;
        let frame = self.current_frame;
        let cmd = self.command_buffers[frame];
        let image_available = self.image_available_semaphores[frame];
        let fence = self.in_flight_fences[frame];
        let mut needs_recreate = false;

        // 1. Wait until this frame slot's previous submission has finished on the GPU. A wait that
        //    cannot confirm completion is unrecoverable — poison the renderer (terminal) rather than
        //    risk reusing resources the GPU may still hold.
        unsafe {
            dev.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| {
                VulkanError::Terminal(format!(
                    "frame fence wait could not confirm completion: {e:?}"
                ))
            })?;
        }

        // 2. Acquire the next swapchain image. image_available is signaled when it's safe to render
        //    into. Out-of-date = the swapchain no longer matches the window: skip this frame and
        //    report it so the renderer recreates. Return BEFORE resetting the fence (it must stay
        //    signaled, or step 1 next time around deadlocks). SUBOPTIMAL still renders but flags a
        //    recreate.
        let image_index = match unsafe {
            swapchain.swapchain_loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        } {
            Ok((index, suboptimal)) => {
                needs_recreate |= suboptimal;
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(FrameOutcome::SwapchainOutOfDate { submitted: false })
            }
            Err(e) => return Err(VulkanError::from(e)),
        };
        // render-finished is per image (see field comment), indexed by the acquired image.
        let render_finished = self.render_finished_semaphores[image_index as usize];

        // 3. Only now that we're committed to submitting, unsignal the fence.
        unsafe {
            dev.reset_fences(&[fence]).map_err(VulkanError::from)?;
        }

        // 3b. Write this frame's camera into its uniform buffer. Safe without extra sync: the buffer is
        //     per-frame-in-flight and step 1's fence wait guarantees the GPU finished the previous
        //     use of this slot.
        // Past the fence reset the fence is unsignaled; any failure now leaves nothing to signal it,
        // so a later frame's wait would hang. Treat every post-reset failure as terminal.
        (|| -> Result<()> {
            uniforms.update(frame, camera)?;
            light.update(frame, directional_light, point_lights)?;
            // One InstanceData per object, in input order, so each run emitted by
            // emit_object_draws below can offset into this frame's buffer via firstInstance and
            // cover its own instanceCount of consecutive entries.
            let instances: Vec<InstanceData> = objects
                .iter()
                .map(|object| {
                    InstanceData::from((
                        object.model,
                        object.specular.color,
                        object.specular.shininess,
                    ))
                })
                .collect();
            instance_buffer.write(frame, &instances)?;
            if let Some(text) = text.as_deref_mut() {
                text.update(frame, text_vertices)?;
            }
            Ok(())
        })()
        .map_err(|e| {
            VulkanError::Terminal(format!("frame update failed after fence reset: {e}"))
        })?;

        // 4. Record the command buffer.
        self.record(
            dev,
            cmd,
            pipeline,
            swapchain,
            depth,
            uniforms.descriptor_set(frame),
            light.descriptor_set(frame),
            instance_buffer.buffer(frame),
            image_index,
            objects,
            owner,
            meshes,
            texture_sets,
            white_set,
            text.as_deref(),
            frame,
            text_vertices.len() as u32,
        )
        .map_err(|e| {
            VulkanError::Terminal(format!("frame recording failed after fence reset: {e}"))
        })?;

        // 5. Submit: wait on image_available at the color-write stage, signal render_finished,
        //    and signal `fence` on completion so step 1 can reclaim this slot next cycle.
        let wait_semaphores = [image_available];
        let wait_stages = [acquire_wait_stage()];
        let command_buffers = [cmd];
        let signal_semaphores = [render_finished];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            dev.queue_submit(device.graphics_queue, &[submit_info], fence)
                .map_err(|e| {
                    VulkanError::Terminal(format!("queue submit failed after fence reset: {e:?}"))
                })?;
        }

        // 6. Present: wait on render_finished so the display engine doesn't show a half-drawn
        //    image. Out-of-date/suboptimal flags a recreate (the renderer handles it next frame).
        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        match unsafe {
            swapchain
                .swapchain_loader
                .queue_present(device.present_queue, &present_info)
        } {
            Ok(suboptimal) => needs_recreate |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                needs_recreate = true;
            }
            Err(e) => return Err(VulkanError::from(e)),
        }

        // 7. Advance to the next frame-in-flight slot.
        self.current_frame = next_frame(frame);

        Ok(if needs_recreate {
            FrameOutcome::SwapchainOutOfDate { submitted: true }
        } else {
            FrameOutcome::Rendered
        })
    }

    /// Records the draw into `cmd` for the acquired `image_index`. With dynamic rendering there are
    /// no subpass dependencies, so the swapchain image's layout transitions are explicit barriers.
    // Command recording genuinely needs all these handles; a mesh-bundle abstraction is a deferred
    // non-goal (specs/003), so allow the arg count rather than introduce one just for clippy.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        dev: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline: &GraphicsPipeline,
        swapchain: &VulkanSwapchain,
        depth: &DepthResources,
        descriptor_set: vk::DescriptorSet,
        light_set: vk::DescriptorSet,
        instance_buffer: vk::Buffer,
        image_index: u32,
        objects: &[Object],
        owner: u64,
        meshes: &[StoredMesh],
        texture_sets: &[vk::DescriptorSet],
        white_set: vk::DescriptorSet,
        text: Option<&TextRenderer>,
        frame: usize,
        text_vertex_count: u32,
    ) -> Result<()> {
        let image = swapchain.images[image_index as usize];
        let image_view = swapchain.image_views[image_index as usize];
        let extent = swapchain.extent;

        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::from)?;
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(VulkanError::from)?;

            // a. UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL before we render into the image.
            let to_color = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                acquire_wait_stage(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_color],
            );

            // Discard contents, but order reuse of the shared depth image after the previous
            // submission's early/late depth accesses (including attachment load/store).
            let to_depth = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(depth.image())
                .subresource_range(depth_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                depth_stages(),
                depth_stages(),
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_depth],
            );

            // b. Begin dynamic rendering into the acquired image view, clearing to dark blue.
            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.08, 1.0],
                },
            };
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear);
            let color_attachments = [color_attachment];
            // Depth attachment: clear to the far plane (1.0) each frame; don't store (nothing reads
            // depth after the pass).
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view())
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);
            dev.cmd_begin_rendering(cmd, &rendering_info);

            // c. Bind the pipeline and the per-frame camera (set = 0) and light (set = 2) descriptor
            //    sets. Both are frame-global; the per-object texture (set = 1) binds in the draw loop.
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline_layout,
                2,
                &[light_set],
                &[],
            );

            // d. Viewport + scissor are dynamic (spec 001), so they must be set every recording.
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            dev.cmd_set_viewport(cmd, 0, &[viewport]);
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            dev.cmd_set_scissor(cmd, 0, &[scissor]);

            // e. Bind the per-frame instance buffer once, then each object's own mesh and texture,
            //    then draw the indexed geometry with that object's firstInstance offset into it.
            emit_object_draws(
                objects,
                owner,
                meshes,
                texture_sets,
                white_set,
                instance_buffer,
                |command| match command {
                    ObjectCommand::BindInstanceBuffer(buffer) => {
                        dev.cmd_bind_vertex_buffers(cmd, 1, &[buffer], &[0]);
                    }
                    ObjectCommand::BindMesh(vertex_buffer, index_buffer) => {
                        dev.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
                        dev.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT16);
                    }
                    ObjectCommand::BindTexture(set) => dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline_layout,
                        1,
                        &[set],
                        &[],
                    ),
                    ObjectCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_instance,
                    } => {
                        dev.cmd_draw_indexed(
                            cmd,
                            index_count,
                            instance_count,
                            0,
                            0,
                            first_instance,
                        );
                    }
                },
            );

            if let Some(text) = text {
                text.record(dev, cmd, frame, text_vertex_count);
            }

            // f. End rendering.
            dev.cmd_end_rendering(cmd);

            // g. COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR so the display engine can show it.
            let to_present = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );

            dev.end_command_buffer(cmd).map_err(VulkanError::from)?;
        }

        Ok(())
    }
}

// One emission path for Vulkan recording and CPU tests; no per-frame command list allocation.
pub(super) enum ObjectCommand {
    BindInstanceBuffer(vk::Buffer),
    BindMesh(vk::Buffer, vk::Buffer),
    BindTexture(vk::DescriptorSet),
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_instance: u32,
    },
}

/// Emit the per-frame instance-buffer bind, then one bind-mesh/bind-texture/draw-indexed sequence
/// per contiguous RUN of objects that share a resolved `(MeshId, resolved-texture-identity)` key —
/// not one per object. `object.mesh`/`object.texture` are resolved here via
/// `mesh::resolve_mesh`/`resolve_texture_set` — both `.expect`-able because the caller
/// (`render_objects_with_text`'s up-front validation pass) already proved every object's handles
/// resolve against `owner`/`meshes`/`texture_sets` before any of this runs.
///
/// The batch key's texture half is the RESOLVED `vk::DescriptorSet` (what `resolve_texture_set`
/// returns), not the raw `Option<TextureId>` the object was described with. This means two objects
/// batch together exactly when they'd bind the identical GPU descriptor set — e.g. two objects with
/// `texture: None` both resolve to `white_set` and merge; an object with `texture: None` and one
/// with `texture: Some(id)` do NOT merge unless `id` happens to resolve to that exact same
/// descriptor set, even if the texture at `id` is itself all-white pixel data. Batching never
/// inspects pixel content — only bound-resource identity — so a visual coincidence (a distinct
/// texture that renders the same color as the white default) can never silently merge two draws
/// that are backed by different GPU resources.
///
/// `instance_buffer` is the SAME buffer for every object this frame (it holds one `InstanceData`
/// per object, written by the caller before recording) — bound once before the per-run loop, not
/// emitted per run, the same way the camera/light descriptor sets are bound once per frame rather
/// than once per object. Each run's draw offsets into it via `first_instance` (the index of the
/// run's first object in `objects`) and covers `instance_count` consecutive instances. Nothing is
/// emitted for an empty `objects` — there is nothing to bind or draw.
pub(super) fn emit_object_draws(
    objects: &[Object],
    owner: u64,
    meshes: &[StoredMesh],
    texture_sets: &[vk::DescriptorSet],
    white_set: vk::DescriptorSet,
    instance_buffer: vk::Buffer,
    mut emit: impl FnMut(ObjectCommand),
) {
    if objects.is_empty() {
        return;
    }
    emit(ObjectCommand::BindInstanceBuffer(instance_buffer));

    let resolve = |object: &Object| -> (mesh::MeshId, &StoredMesh, vk::DescriptorSet) {
        let resolved = mesh::resolve_mesh(object.mesh, owner, meshes)
            .expect("object handles were validated above, before this point was reached");
        let set = resolve_texture_set(object.texture, owner, texture_sets, white_set)
            .expect("object handles were validated above, before this point was reached");
        (object.mesh, resolved, set)
    };

    let mut run_start = 0usize;
    let (mut run_mesh_id, mut run_mesh, mut run_set) = resolve(&objects[0]);
    for (index, object) in objects.iter().enumerate().skip(1) {
        let (mesh_id, mesh, set) = resolve(object);
        if mesh_id != run_mesh_id || set != run_set {
            emit_run(&mut emit, run_mesh, run_set, run_start, index - run_start);
            run_start = index;
            run_mesh_id = mesh_id;
            run_mesh = mesh;
            run_set = set;
        }
    }
    emit_run(
        &mut emit,
        run_mesh,
        run_set,
        run_start,
        objects.len() - run_start,
    );
}

/// Emit one run's bind-mesh/bind-texture/draw-indexed sequence: `length` consecutive instances
/// starting at `first_instance`, all drawing `mesh`'s geometry with `set` bound.
fn emit_run(
    emit: &mut impl FnMut(ObjectCommand),
    mesh: &StoredMesh,
    set: vk::DescriptorSet,
    first_instance: usize,
    length: usize,
) {
    emit(ObjectCommand::BindMesh(
        mesh.vertex_buffer.buffer,
        mesh.index_buffer.buffer,
    ));
    emit(ObjectCommand::BindTexture(set));
    emit(ObjectCommand::DrawIndexed {
        index_count: mesh.index_count,
        instance_count: length as u32,
        first_instance: first_instance as u32,
    });
}

// The image transition's source scope must chain with the acquire semaphore wait.
fn acquire_wait_stage() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
}

fn depth_stages() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
}

fn next_frame(current: usize) -> usize {
    (current + 1) % FRAMES_IN_FLIGHT
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn depth_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

impl Drop for FrameContext {
    fn drop(&mut self) {
        // Fences, then semaphores, then the command pool (which frees its command buffers
        // implicitly -- no separate free_command_buffers call needed).
        unsafe {
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::buffer::Buffer;
    use super::super::texture::SpecularMaterial;
    use super::*;
    use ash::vk::Handle;

    #[derive(Debug, PartialEq)]
    enum RecordedCommand {
        InstanceBuffer(vk::Buffer),
        Mesh(vk::Buffer, vk::Buffer),
        Bind(vk::DescriptorSet),
        Draw {
            index_count: u32,
            instance_count: u32,
            first_instance: u32,
        },
    }

    fn set(raw: u64) -> vk::DescriptorSet {
        vk::DescriptorSet::from_raw(raw)
    }

    /// A CPU-only `StoredMesh` fixture: `vertex_raw`/`index_raw` become its bind-call handles (so
    /// tests can tell which mesh a `BindMesh` command came from), `index_count` its draw count.
    fn stored_mesh(vertex_raw: u64, index_raw: u64, index_count: u32) -> StoredMesh {
        StoredMesh {
            vertex_buffer: Buffer::test_fixture(vk::Buffer::from_raw(vertex_raw)),
            index_buffer: Buffer::test_fixture(vk::Buffer::from_raw(index_raw)),
            index_count,
        }
    }

    fn record_objects(
        objects: &[Object],
        owner: u64,
        meshes: &[StoredMesh],
        sets: &[vk::DescriptorSet],
        white: vk::DescriptorSet,
        instance_buffer: vk::Buffer,
    ) -> Vec<RecordedCommand> {
        let mut commands = Vec::new();
        emit_object_draws(
            objects,
            owner,
            meshes,
            sets,
            white,
            instance_buffer,
            |command| {
                commands.push(match command {
                    ObjectCommand::BindInstanceBuffer(buffer) => {
                        RecordedCommand::InstanceBuffer(buffer)
                    }
                    ObjectCommand::BindMesh(vertex, index) => RecordedCommand::Mesh(vertex, index),
                    ObjectCommand::BindTexture(s) => RecordedCommand::Bind(s),
                    ObjectCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_instance,
                    } => RecordedCommand::Draw {
                        index_count,
                        instance_count,
                        first_instance,
                    },
                })
            },
        );
        commands
    }

    fn instances(raw: u64) -> vk::Buffer {
        vk::Buffer::from_raw(raw)
    }

    #[test]
    fn empty_objects_emit_no_commands() {
        let owner = crate::renderer::handle::next_owner_id();
        assert!(
            record_objects(&[], owner, &[], &[set(0xA0)], set(0x10), instances(0xC0)).is_empty()
        );
    }

    #[test]
    fn interleaved_objects_bind_each_objects_own_mesh_texture_and_model_in_order() {
        let white = set(0x10);
        let sets = [set(0xA0), set(0xB0)];
        let owner = crate::renderer::handle::next_owner_id();
        // Two distinct meshes with different index counts, so a wrong bind/count pairing would be
        // visible in the asserted command sequence below.
        let meshes = [
            stored_mesh(0x1000, 0x1001, 3),
            stored_mesh(0x2000, 0x2001, 6),
        ];
        let mesh_a = mesh::MeshId::new(owner, 0);
        let mesh_b = mesh::MeshId::new(owner, 1);

        let m0 = crate::math::translate(&crate::math::Vec3::new(1.0, 0.0, 0.0));
        let m1 = crate::math::rotate(&crate::math::Vec3::new(0.0, 1.0, 0.0), 0.5);
        let m2 = crate::math::translate(&crate::math::Vec3::new(-1.0, 0.0, 0.0));
        let specular0 = SpecularMaterial::default();
        let specular1 = SpecularMaterial {
            color: crate::math::Vec3::new(1.0, 1.0, 1.0),
            shininess: 64.0,
        };
        let specular2 = SpecularMaterial {
            color: crate::math::Vec3::new(0.5, 0.0, 0.5),
            shininess: 8.0,
        };
        // mesh A, then mesh B, then mesh A again: proves each draw resolves ITS OWN object's mesh
        // rather than reusing whatever the previous object bound.
        let objects = [
            Object {
                mesh: mesh_a,
                model: m0,
                texture: None,
                specular: specular0,
            },
            Object {
                mesh: mesh_b,
                model: m1,
                texture: Some(crate::renderer::texture::TextureId::new(owner, 1)),
                specular: specular1,
            },
            Object {
                mesh: mesh_a,
                model: m2,
                texture: None,
                specular: specular2,
            },
        ];
        let instance_buffer = instances(0xD0);
        assert_eq!(
            record_objects(&objects, owner, &meshes, &sets, white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 0,
                },
                RecordedCommand::Mesh(
                    meshes[1].vertex_buffer.buffer,
                    meshes[1].index_buffer.buffer
                ),
                RecordedCommand::Bind(sets[1]),
                RecordedCommand::Draw {
                    index_count: 6,
                    instance_count: 1,
                    first_instance: 1,
                },
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 2,
                },
            ]
        );
    }

    /// Builds `count` objects all sharing `mesh` and `texture`, with distinct (but otherwise
    /// irrelevant) transforms so each is a genuinely separate object rather than a clone.
    fn objects_sharing(
        mesh: mesh::MeshId,
        texture: Option<crate::renderer::texture::TextureId>,
        count: usize,
    ) -> Vec<Object> {
        (0..count)
            .map(|i| Object {
                mesh,
                model: crate::math::translate(&crate::math::Vec3::new(i as f32, 0.0, 0.0)),
                texture,
                specular: SpecularMaterial::default(),
            })
            .collect()
    }

    #[test]
    fn single_object_is_one_run_of_length_one() {
        let white = set(0x10);
        let owner = crate::renderer::handle::next_owner_id();
        let meshes = [stored_mesh(0x1000, 0x1001, 3)];
        let mesh_a = mesh::MeshId::new(owner, 0);
        let objects = objects_sharing(mesh_a, None, 1);
        let instance_buffer = instances(0xD0);
        assert_eq!(
            record_objects(&objects, owner, &meshes, &[], white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 0,
                },
            ]
        );
    }

    #[test]
    fn all_objects_sharing_mesh_and_texture_merge_into_one_run() {
        let white = set(0x10);
        let owner = crate::renderer::handle::next_owner_id();
        let meshes = [stored_mesh(0x1000, 0x1001, 3)];
        let mesh_a = mesh::MeshId::new(owner, 0);
        let objects = objects_sharing(mesh_a, None, 5);
        let instance_buffer = instances(0xD0);
        // Exactly one bind-mesh/bind-texture/draw sequence, with instance_count == objects.len(),
        // not one per object.
        assert_eq!(
            record_objects(&objects, owner, &meshes, &[], white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 5,
                    first_instance: 0,
                },
            ]
        );
    }

    #[test]
    fn strictly_alternating_mesh_produces_one_run_per_object() {
        let white = set(0x10);
        let owner = crate::renderer::handle::next_owner_id();
        let meshes = [
            stored_mesh(0x1000, 0x1001, 3),
            stored_mesh(0x2000, 0x2001, 6),
        ];
        let mesh_a = mesh::MeshId::new(owner, 0);
        let mesh_b = mesh::MeshId::new(owner, 1);
        let objects: Vec<Object> = [mesh_a, mesh_b, mesh_a, mesh_b]
            .into_iter()
            .enumerate()
            .map(|(i, mesh)| Object {
                mesh,
                model: crate::math::translate(&crate::math::Vec3::new(i as f32, 0.0, 0.0)),
                texture: None,
                specular: SpecularMaterial::default(),
            })
            .collect();
        let instance_buffer = instances(0xD0);
        assert_eq!(
            record_objects(&objects, owner, &meshes, &[], white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 0,
                },
                RecordedCommand::Mesh(
                    meshes[1].vertex_buffer.buffer,
                    meshes[1].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 6,
                    instance_count: 1,
                    first_instance: 1,
                },
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 2,
                },
                RecordedCommand::Mesh(
                    meshes[1].vertex_buffer.buffer,
                    meshes[1].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 6,
                    instance_count: 1,
                    first_instance: 3,
                },
            ]
        );
    }

    #[test]
    fn a_a_b_b_a_produces_three_runs_in_exact_order() {
        let white = set(0x10);
        let owner = crate::renderer::handle::next_owner_id();
        let meshes = [
            stored_mesh(0x1000, 0x1001, 3),
            stored_mesh(0x2000, 0x2001, 6),
        ];
        let mesh_a = mesh::MeshId::new(owner, 0);
        let mesh_b = mesh::MeshId::new(owner, 1);
        let objects: Vec<Object> = [mesh_a, mesh_a, mesh_b, mesh_b, mesh_a]
            .into_iter()
            .enumerate()
            .map(|(i, mesh)| Object {
                mesh,
                model: crate::math::translate(&crate::math::Vec3::new(i as f32, 0.0, 0.0)),
                texture: None,
                specular: SpecularMaterial::default(),
            })
            .collect();
        let instance_buffer = instances(0xD0);
        assert_eq!(
            record_objects(&objects, owner, &meshes, &[], white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                // Run 1: objects[0..2), mesh A, length 2, first_instance 0.
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 2,
                    first_instance: 0,
                },
                // Run 2: objects[2..4), mesh B, length 2, first_instance 2.
                RecordedCommand::Mesh(
                    meshes[1].vertex_buffer.buffer,
                    meshes[1].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 6,
                    instance_count: 2,
                    first_instance: 2,
                },
                // Run 3: objects[4..5), mesh A again, length 1, first_instance 4.
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 4,
                },
            ]
        );
    }

    #[test]
    fn batching_key_is_resolved_descriptor_set_identity_not_option_shape() {
        let white = set(0x10);
        let owner = crate::renderer::handle::next_owner_id();
        let meshes = [stored_mesh(0x1000, 0x1001, 3)];
        let mesh_a = mesh::MeshId::new(owner, 0);
        // `sets[0]` is deliberately the SAME descriptor set as `white`; `sets[1]` is a distinct
        // one. This is contrived (a real texture store would never alias the white default), but
        // it isolates the rule under test: batching compares the RESOLVED `vk::DescriptorSet`,
        // never whether the object described its texture as `None` or `Some`.
        let sets = [white, set(0xB0)];
        let tex_same_as_white = Some(crate::renderer::texture::TextureId::new(owner, 0));
        let tex_distinct = Some(crate::renderer::texture::TextureId::new(owner, 1));

        // `None` (-> white) and `Some(id)` that resolves to that SAME set: one run of length 2.
        let objects = [
            Object {
                mesh: mesh_a,
                model: crate::math::translate(&crate::math::Vec3::new(0.0, 0.0, 0.0)),
                texture: None,
                specular: SpecularMaterial::default(),
            },
            Object {
                mesh: mesh_a,
                model: crate::math::translate(&crate::math::Vec3::new(1.0, 0.0, 0.0)),
                texture: tex_same_as_white,
                specular: SpecularMaterial::default(),
            },
        ];
        let instance_buffer = instances(0xD0);
        assert_eq!(
            record_objects(&objects, owner, &meshes, &sets, white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 2,
                    first_instance: 0,
                },
            ]
        );

        // `None` (-> white) and `Some(id)` that resolves to a DIFFERENT set: two separate runs,
        // even though the distinct texture might render visually identical to white -- batching
        // never inspects pixel content, only bound-resource identity.
        let objects = [
            Object {
                mesh: mesh_a,
                model: crate::math::translate(&crate::math::Vec3::new(0.0, 0.0, 0.0)),
                texture: None,
                specular: SpecularMaterial::default(),
            },
            Object {
                mesh: mesh_a,
                model: crate::math::translate(&crate::math::Vec3::new(1.0, 0.0, 0.0)),
                texture: tex_distinct,
                specular: SpecularMaterial::default(),
            },
        ];
        assert_eq!(
            record_objects(&objects, owner, &meshes, &sets, white, instance_buffer),
            vec![
                RecordedCommand::InstanceBuffer(instance_buffer),
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(white),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 0,
                },
                RecordedCommand::Mesh(
                    meshes[0].vertex_buffer.buffer,
                    meshes[0].index_buffer.buffer
                ),
                RecordedCommand::Bind(sets[1]),
                RecordedCommand::Draw {
                    index_count: 3,
                    instance_count: 1,
                    first_instance: 1,
                },
            ]
        );
    }

    #[test]
    fn out_of_date_acquire_does_not_count_as_submission() {
        assert!(!FrameOutcome::SwapchainOutOfDate { submitted: false }.submitted());
        assert!(FrameOutcome::SwapchainOutOfDate { submitted: true }.submitted());
        assert!(FrameOutcome::Rendered.submitted());
    }

    #[test]
    fn acquire_wait_orders_the_color_layout_transition() {
        assert_eq!(
            acquire_wait_stage(),
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        );
    }

    #[test]
    fn shared_depth_dependency_covers_early_and_late_accesses() {
        assert!(depth_stages().contains(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS));
        assert!(depth_stages().contains(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS));
        assert!(!depth_stages().contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT));
    }

    #[test]
    fn next_frame_wraps_around() {
        assert_eq!(next_frame(0), 1);
        assert_eq!(next_frame(FRAMES_IN_FLIGHT - 1), 0);
    }
}
