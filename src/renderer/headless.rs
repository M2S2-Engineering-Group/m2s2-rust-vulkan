// Offscreen color render target for headless (no-surface) rendering in GPU tests. Stands in for a
// swapchain image: no surface, no present — just a plain color `VkImage` the one-shot recorder
// (Task 3) renders into and reads back from. See specs/019-headless-rendering-and-golden-image-tests.md.

use crate::error::{Result, VulkanError};
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::{Buffer, Index, Vertex};
use crate::renderer::command::{self, ObjectCommand};
use crate::renderer::depth::DepthResources;
use crate::renderer::device::VulkanDevice;
use crate::renderer::handle;
use crate::renderer::instance::VulkanInstance;
use crate::renderer::instance_buffer::InstanceBuffer;
use crate::renderer::light::{DirectionalLight, Light, PointLight};
use crate::renderer::mesh::{self, MeshId, StoredMesh};
use crate::renderer::pipeline::{GraphicsPipeline, InstanceData};
use crate::renderer::scene::Object;
use crate::renderer::submission::{SubmitState, SubmitStep};
use crate::renderer::text::{self, TextDraw, TextRenderer, TextVertex};
use crate::renderer::texture::{self, StoredTexture, TextureId, TextureOptions};
use crate::renderer::uniform::{self, Uniforms};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::mem::ManuallyDrop;

/// Color formats in preference order: plain RGBA8 first, then the swapchain-typical BGRA8 as a
/// fallback.
pub(crate) const OFFSCREEN_COLOR_FORMAT_CANDIDATES: [vk::Format; 2] =
    [vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM];

/// A plain color image + view standing in for a swapchain image in headless tests, plus a
/// host-visible readback buffer for pulling rendered pixels back to the CPU. Device-local, sized to
/// `extent` at construction (not recreated on resize — headless tests have no resize). Not `Drop`:
/// freeing its allocation (and the readback buffer's) needs `&mut Allocator`, so call `destroy`
/// explicitly.
pub(crate) struct OffscreenTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    readback: Buffer,
    extent: vk::Extent2D,
    format: vk::Format,
    has_rendered: bool,
}

impl OffscreenTarget {
    pub(crate) fn new(
        device: &VulkanDevice,
        instance: &VulkanInstance,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let candidates: Vec<(vk::Format, vk::FormatFeatureFlags)> =
            OFFSCREEN_COLOR_FORMAT_CANDIDATES
                .iter()
                .map(|&format| {
                    let props = unsafe {
                        instance
                            .instance
                            .get_physical_device_format_properties(device.physical_device, format)
                    };
                    (format, props.optimal_tiling_features)
                })
                .collect();
        let format = select_color_format(&candidates).ok_or_else(|| {
            VulkanError::InitializationError(
                "no supported offscreen color-attachment format".into(),
            )
        })?;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe {
            device
                .device
                .create_image(&image_info, None)
                .map_err(VulkanError::from)?
        };

        let requirements = unsafe { device.device.get_image_memory_requirements(image) };
        let allocation_result = allocator.allocate(&AllocationCreateDesc {
            name: "offscreen-color",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false, // OPTIMAL-tiled image
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation_result {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.device.destroy_image(image, None) };
                return Err(error);
            }
        };
        let result = (|| {
            unsafe {
                device
                    .device
                    .bind_image_memory(image, allocation.memory(), allocation.offset())
                    .map_err(VulkanError::from)?;
            }

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            let view = unsafe {
                device
                    .device
                    .create_image_view(&view_info, None)
                    .map_err(VulkanError::from)?
            };

            Ok(view)
        })();
        let view = match result {
            Ok(view) => view,
            Err(error) => {
                unsafe { device.device.destroy_image(image, None) };
                let _ = allocator.free(allocation);
                return Err(error);
            }
        };

        let readback_result = Buffer::new_host_visible_mapped(
            device,
            allocator,
            extent.width as u64 * extent.height as u64 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        let readback = match readback_result {
            Ok(readback) => readback,
            Err(error) => {
                unsafe {
                    device.device.destroy_image_view(view, None);
                    device.device.destroy_image(image, None);
                }
                let _ = allocator.free(allocation);
                return Err(error);
            }
        };

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            readback,
            extent,
            format,
            has_rendered: false,
        })
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }

    pub(crate) fn has_rendered(&self) -> bool {
        self.has_rendered
    }

    pub(crate) fn mark_rendered(&mut self) {
        self.has_rendered = true;
    }

    /// Destroy the view + image (via `device`) and free the image allocation and the readback
    /// buffer (via `allocator`). Must run before the `Allocator` is dropped.
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe {
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation {
            let _ = allocator.free(allocation);
        }
        self.readback.destroy(device, allocator);
    }
}

/// Picks the color format from `candidates`, which the caller supplies in priority order paired
/// with each format's queried `optimal_tiling_features`. Returns the first format whose features
/// advertise `COLOR_ATTACHMENT` (i.e. it's usable as a color attachment with OPTIMAL tiling), or
/// `None` if none qualify. Pure given the queried properties, so it's testable without a device.
fn select_color_format(candidates: &[(vk::Format, vk::FormatFeatureFlags)]) -> Option<vk::Format> {
    candidates
        .iter()
        .find(|(_, features)| features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT))
        .map(|&(format, _)| format)
}

/// A self-contained Vulkan renderer with no swapchain/window: it owns a headless instance/device and
/// the same device/allocator/pipeline/depth/uniform resource graph `VulkanRenderer` owns, but draws
/// into an `OffscreenTarget` via one-shot command recording instead of acquiring/presenting a
/// swapchain image. For golden-image/determinism GPU tests (specs/019); not used outside `#[cfg(test)]`.
///
/// `instance`/`device`/`pipeline` are `ManuallyDrop` (and the rest `Option`) for the same reason as
/// `VulkanRenderer`: `Drop::drop` only has `&mut self`, but tearing them down needs to consume them by
/// value (`GraphicsPipeline`'s own `Drop`, or an explicit `destroy(self, ..)`) in a specific order, not
/// the struct's declaration order Rust's default field-drop glue would otherwise use.
pub(crate) struct HeadlessRenderer {
    device: ManuallyDrop<VulkanDevice>,
    instance: ManuallyDrop<VulkanInstance>,
    allocator: ManuallyDrop<Allocator>,
    // Append-only store resolved by the same command path as VulkanRenderer.
    meshes: Vec<StoredMesh>,
    owner: u64,
    depth: Option<DepthResources>,
    uniforms: Option<Uniforms>,
    texture_set_layout: vk::DescriptorSetLayout,
    white_texture: Option<StoredTexture>,
    textures: Vec<StoredTexture>,
    texture_sets: Vec<vk::DescriptorSet>,
    light: Option<Light>,
    instance_buffer: Option<InstanceBuffer>,
    pipeline: ManuallyDrop<GraphicsPipeline>,
    target: Option<OffscreenTarget>,
    text: Option<TextRenderer>,
    text_vertices: Vec<TextVertex>,
    terminal: bool,
    // Counters from the most recently *completed* `render_frame_with_text` call, re-derived by
    // walking the same `emit_object_draws` run-grouping `VulkanRenderer::draw_frame` counts with
    // (see `mod.rs`'s matching comment) -- not a second, parallel batching implementation. Let
    // golden-image tests assert the real draw-count reduction instancing exists for, not just a
    // visual match.
    scene_draw_calls: usize,
    scene_instances: usize,
}

impl HeadlessRenderer {
    /// Mirrors `VulkanRenderer::init`'s resource-construction order (spec 012's `Option`-held-locals
    /// rollback-on-error pattern), substituting the headless instance/device and an `OffscreenTarget`
    /// for the swapchain. `extent` sizes the depth attachment and the offscreen color target alike.
    pub(crate) fn new(extent: (u32, u32), vertices: &[Vertex], indices: &[Index]) -> Result<Self> {
        let mut renderer = Self::new_empty(extent)?;
        renderer.create_mesh(vertices, indices)?;
        Ok(renderer)
    }

    pub(crate) fn new_empty(extent: (u32, u32)) -> Result<Self> {
        let instance = VulkanInstance::new_headless()?;
        let device = VulkanDevice::new_headless(&instance)?;
        let mut allocator = Allocator::new(&instance, &device)?;

        let extent_2d = vk::Extent2D {
            width: extent.0,
            height: extent.1,
        };

        let mut depth = None;
        let mut uniforms = None;
        let mut texture_set_layout = None;
        let mut white_texture = None;
        let mut light = None;
        let mut instance_buffer = None;
        let mut target = None;
        let resources = (|| {
            depth = Some(DepthResources::new(
                &instance,
                &device,
                &mut allocator,
                extent_2d,
            )?);
            uniforms = Some(Uniforms::new(&device, &mut allocator)?);

            let layout = texture::create_set_layout(&device)?;
            texture_set_layout = Some(layout);
            white_texture = Some(StoredTexture::new(
                &device,
                &mut allocator,
                &instance,
                layout,
                &[255, 255, 255, 255],
                1,
                1,
                TextureOptions::default(),
            )?);

            light = Some(Light::new(&device, &mut allocator)?);
            // One slot: the headless renderer has no frames-in-flight concept (every render_frame
            // is a one-shot submission that waits for its own fence), always at frame index 0 --
            // same reasoning as its existing single-slot Uniforms/Light usage.
            instance_buffer = Some(InstanceBuffer::new(&device, &mut allocator, 1)?);

            // Built before the pipeline (despite being stored in the struct after it) because the
            // pipeline needs the target's chosen color format as a construction argument.
            target = Some(OffscreenTarget::new(
                &device,
                &instance,
                &mut allocator,
                extent_2d,
            )?);

            let pipeline = GraphicsPipeline::new(
                &device,
                target.as_ref().unwrap().format(),
                depth.as_ref().unwrap().format(),
                &[
                    uniforms.as_ref().unwrap().descriptor_set_layout(),
                    layout,
                    light.as_ref().unwrap().descriptor_set_layout(),
                ],
            )?;

            Ok(pipeline)
        })();
        let pipeline = match resources {
            Ok(pipeline) => pipeline,
            Err(error) => {
                if let Some(resource) = target.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = instance_buffer.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = light.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = white_texture.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(layout) = texture_set_layout.take() {
                    unsafe { device.device.destroy_descriptor_set_layout(layout, None) };
                }
                if let Some(resource) = uniforms.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = depth.take() {
                    resource.destroy(&device, &mut allocator);
                }
                return Err(error);
            }
        };

        Ok(Self {
            device: ManuallyDrop::new(device),
            instance: ManuallyDrop::new(instance),
            allocator: ManuallyDrop::new(allocator),
            meshes: Vec::new(),
            owner: handle::next_owner_id(),
            depth,
            uniforms,
            texture_set_layout: texture_set_layout.unwrap(),
            white_texture,
            textures: Vec::new(),
            texture_sets: Vec::new(),
            light,
            instance_buffer,
            pipeline: ManuallyDrop::new(pipeline),
            target,
            text: None,
            text_vertices: Vec::new(),
            terminal: false,
            scene_draw_calls: 0,
            scene_instances: 0,
        })
    }

    /// Records a one-shot command buffer that draws `objects` under `camera` into the owned
    /// `OffscreenTarget`, submits and waits on a transient fence, and reads the rendered pixels back
    /// to the CPU as tightly-packed RGBA8 (regardless of the target's underlying Vulkan format — BGRA8
    /// is swizzled to RGBA8 before returning). Frame-slot 0 always, since there is no
    /// frames-in-flight concept for a one-shot renderer. Calling this repeatedly is safe: the color
    /// image's layout transition picks its source scope based on whether a previous call already left
    /// it in `TRANSFER_SRC_OPTIMAL`.
    pub(crate) fn render_frame(
        &mut self,
        objects: &[Object],
        camera: uniform::Camera,
        directional_light: &DirectionalLight,
        point_lights: &[PointLight],
    ) -> Result<Vec<u8>> {
        self.render_frame_with_text(objects, camera, directional_light, point_lights, &[])
    }

    pub(crate) fn render_frame_with_text(
        &mut self,
        objects: &[Object],
        camera: uniform::Camera,
        directional_light: &DirectionalLight,
        point_lights: &[PointLight],
        draws: &[TextDraw<'_>],
    ) -> Result<Vec<u8>> {
        if self.terminal {
            return Err(VulkanError::Terminal(
                "headless renderer is terminal".into(),
            ));
        }
        uniform::validate_camera(&camera)?;
        for object in objects {
            texture::validate_specular(&object.specular)?;
            mesh::resolve_mesh(object.mesh, self.owner, &self.meshes)?;
            texture::resolve_texture_set(
                object.texture,
                self.owner,
                &self.texture_sets,
                self.white_texture.as_ref().unwrap().set,
            )?;
        }
        let target = self.target.as_ref().unwrap();
        text::layout_text(draws, target.extent(), &mut self.text_vertices)?;
        if self.text.is_none() && !self.text_vertices.is_empty() {
            let result = TextRenderer::new(
                &self.device,
                &mut self.allocator,
                target.format(),
                self.depth.as_ref().unwrap().format(),
            );
            match result {
                Ok(text) => self.text = Some(text),
                Err(error) => {
                    if matches!(error, VulkanError::Terminal(_)) {
                        self.terminal = true;
                    }
                    return Err(error);
                }
            }
        }
        if let Some(text) = self.text.as_mut() {
            text.update(0, &self.text_vertices)?;
        }
        self.uniforms
            .as_mut()
            .expect("uniforms present outside teardown")
            .update(0, &uniform::CameraUniforms::from(&camera))?;
        self.light
            .as_mut()
            .expect("light present outside teardown")
            .update(0, directional_light, point_lights)?;
        // One InstanceData per object, in input order, written to slot 0 (the only slot a
        // one-shot headless renderer has) -- see command.rs's draw_frame for the equivalent
        // per-frame-in-flight version this mirrors.
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
        self.instance_buffer
            .as_mut()
            .expect("instance buffer present outside teardown")
            .write(0, &instances)?;

        let dev = &self.device.device;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe {
            dev.create_command_pool(&pool_info, None)
                .map_err(VulkanError::from)?
        };
        let pool_cleanup =
            super::cleanup::Cleanup::new(|| unsafe { dev.destroy_command_pool(pool, None) });

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            dev.allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::from)?
        }[0];

        let target = self
            .target
            .as_ref()
            .expect("target present outside teardown");
        let depth = self.depth.as_ref().expect("depth present outside teardown");
        let extent = target.extent();
        let color_image = target.image;
        let color_view = target.view;

        // First call ever: the image starts UNDEFINED (as created). Every later call: the previous
        // render_frame left it in TRANSFER_SRC_OPTIMAL (after its own readback copy), so the barrier's
        // source scope must match that, not UNDEFINED.
        let (old_layout, src_access, src_stage) = if target.has_rendered() {
            (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
            )
        } else {
            (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            )
        };

        unsafe {
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(VulkanError::from)?;

            let to_color = vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_image)
                .subresource_range(color_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                src_stage,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_color],
            );

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

            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.08, 1.0],
                },
            };
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear);
            let color_attachments = [color_attachment];
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

            let pipeline_handle = self.pipeline.pipeline;
            let pipeline_layout = self.pipeline.pipeline_layout;
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_handle);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[self
                    .uniforms
                    .as_ref()
                    .expect("uniforms present outside teardown")
                    .descriptor_set(0)],
                &[],
            );
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                2,
                &[self
                    .light
                    .as_ref()
                    .expect("light present outside teardown")
                    .descriptor_set(0)],
                &[],
            );

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

            let white_set = self
                .white_texture
                .as_ref()
                .expect("white texture present outside teardown")
                .set;
            let instance_buffer_handle = self
                .instance_buffer
                .as_ref()
                .expect("instance buffer present outside teardown")
                .buffer(0);
            command::emit_object_draws(
                objects,
                self.owner,
                &self.meshes,
                &self.texture_sets,
                white_set,
                instance_buffer_handle,
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
                        pipeline_layout,
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

            if let Some(text) = self.text.as_ref() {
                text.record(dev, cmd, 0, self.text_vertices.len() as u32);
            }
            dev.cmd_end_rendering(cmd);

            let to_transfer = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_image)
                .subresource_range(color_subresource_range());
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );

            let copy = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            dev.cmd_copy_image_to_buffer(
                cmd,
                color_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target.readback.buffer,
                &[copy],
            );

            dev.end_command_buffer(cmd).map_err(VulkanError::from)?;
        }

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let fence = unsafe {
            dev.create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(VulkanError::from)?
        };
        let fence_cleanup =
            super::cleanup::Cleanup::new(|| unsafe { dev.destroy_fence(fence, None) });
        unsafe {
            dev.queue_submit(self.device.graphics_queue, &[submit], fence)
                .map_err(VulkanError::from)?;
        }
        let pending = SubmitState::NotSubmitted.advance(SubmitStep::Submitted);
        let wait = unsafe { dev.wait_for_fences(&[fence], true, u64::MAX) };
        let step = if wait.is_ok() {
            SubmitStep::WaitOk
        } else {
            SubmitStep::WaitFailed
        };
        if !pending.advance(step).may_release_pending() {
            // Wait could not confirm completion: the GPU may still own the pool/fence/target/readback
            // buffer. Leak the pool/fence rather than destroy them, and report terminal -- never free
            // possibly-in-use resources (mirrors upload::submit_once).
            pool_cleanup.disarm();
            fence_cleanup.disarm();
            self.terminal = true;
            return Err(VulkanError::Terminal(format!(
                "headless render fence wait could not confirm completion: {:?}",
                wait.err()
            )));
        }

        let byte_len = extent.width as usize * extent.height as usize * 4;
        let mut pixels =
            target.readback.mapped_bytes().ok_or_else(|| {
                VulkanError::ValidationError("readback buffer is not mapped".into())
            })?[..byte_len]
                .to_vec();
        if target.format() == vk::Format::B8G8R8A8_UNORM {
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }

        self.target
            .as_mut()
            .expect("target present outside teardown")
            .mark_rendered();

        // Re-walk the actual runs `emit_object_draws` grouped `objects` into (not a second,
        // parallel batching implementation) to derive both counters consistently -- mirrors
        // `VulkanRenderer::draw_frame`'s matching comment in `mod.rs`. `vk::Buffer::null()` stands
        // in for the instance buffer: this pass only counts commands, it never records any of
        // them. Every object handle was already proven to resolve by the validation pass above the
        // real recording, so this cannot fail.
        let mut draw_calls = 0usize;
        let mut instances = 0usize;
        let white_set = self
            .white_texture
            .as_ref()
            .expect("white texture present outside teardown")
            .set;
        command::emit_object_draws(
            objects,
            self.owner,
            &self.meshes,
            &self.texture_sets,
            white_set,
            vk::Buffer::null(),
            |command| {
                if let ObjectCommand::DrawIndexed { instance_count, .. } = command {
                    draw_calls += 1;
                    instances += instance_count as usize;
                }
            },
        );
        self.scene_draw_calls = draw_calls;
        self.scene_instances = instances;

        Ok(pixels)
    }

    /// Forwards to the underlying headless instance's validation-error counter (see
    /// `VulkanInstance::validation_error_count`), so tests can assert "zero validation errors"
    /// without reaching into private fields.
    pub(crate) fn validation_error_count(&self) -> u32 {
        self.instance.validation_error_count()
    }

    /// Draw-call count from the most recently completed render: one per run of consecutive objects
    /// sharing a resolved mesh+texture (see `command::emit_object_draws`), not one per object. Lets
    /// tests assert the batching reduction directly instead of inferring it from pixels alone.
    pub(crate) fn scene_draw_calls(&self) -> usize {
        self.scene_draw_calls
    }

    /// Total object/instance count across all runs from the most recently completed render.
    pub(crate) fn scene_instances(&self) -> usize {
        self.scene_instances
    }

    /// Upload additional geometry using the production mesh uploader.
    pub(crate) fn create_mesh(&mut self, vertices: &[Vertex], indices: &[Index]) -> Result<MeshId> {
        if self.terminal {
            return Err(VulkanError::Terminal(
                "headless renderer is terminal".into(),
            ));
        }
        let mesh = match mesh::upload_mesh(&self.device, &mut self.allocator, vertices, indices) {
            Ok(mesh) => mesh,
            Err(error) => {
                if matches!(error, VulkanError::Terminal(_)) {
                    self.terminal = true;
                }
                return Err(error);
            }
        };
        let id = MeshId::new(self.owner, self.meshes.len());
        self.meshes.push(mesh);
        Ok(id)
    }

    /// Upload an RGBA8 image into a new renderer-owned texture using the production uploader.
    /// Single-level (`Mipmaps::Disabled`) — equivalent to
    /// `create_texture_with_options(rgba8, width, height, TextureOptions::default())`.
    /// `#[allow(dead_code)]`: consumed by Task 5's GPU tests (spec 014), which render against a
    /// real, non-white texture. Exercised today only indirectly, via `create_texture_with_options`.
    #[allow(dead_code)]
    pub(crate) fn create_texture(
        &mut self,
        rgba8: &[u8],
        width: u32,
        height: u32,
    ) -> Result<TextureId> {
        self.create_texture_with_options(rgba8, width, height, TextureOptions::default())
    }

    /// Upload an RGBA8 image into a new renderer-owned texture with explicit `options` (e.g.
    /// `Mipmaps::Generate` to generate a full mip chain). Mirrors
    /// `VulkanRenderer::create_texture_with_options`.
    /// `#[allow(dead_code)]`: consumed by Task 5's GPU tests (spec 014).
    #[allow(dead_code)]
    pub(crate) fn create_texture_with_options(
        &mut self,
        rgba8: &[u8],
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> Result<TextureId> {
        if self.terminal {
            return Err(VulkanError::Terminal(
                "headless renderer is terminal".into(),
            ));
        }
        let stored = match StoredTexture::new(
            &self.device,
            &mut self.allocator,
            &self.instance,
            self.texture_set_layout,
            rgba8,
            width,
            height,
            options,
        ) {
            Ok(stored) => stored,
            Err(error) => {
                if matches!(error, VulkanError::Terminal(_)) {
                    self.terminal = true;
                }
                return Err(error);
            }
        };
        let id = TextureId::new(self.owner, self.textures.len());
        self.texture_sets.push(stored.set);
        self.textures.push(stored);
        Ok(id)
    }

    /// Existing single-mesh fixtures keep their original handle and pixels.
    pub(crate) fn mesh(&self) -> MeshId {
        assert!(!self.meshes.is_empty(), "no mesh was uploaded");
        MeshId::new(self.owner, 0)
    }

    /// The raw `vk::Image` behind a stored texture, resolved against this renderer's identity the
    /// same way `render_frame` resolves descriptor sets. Test-only plumbing for
    /// `renderer::mip_readback`'s GPU readback helper (spec 014), which needs the underlying image
    /// handle (not a descriptor set) to copy a specific mip level back to the CPU.
    pub(crate) fn texture_image(&self, id: TextureId) -> vk::Image {
        texture::resolve_texture_image(id, self.owner, &self.textures)
            .expect("texture id must resolve against this renderer")
    }
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

fn depth_stages() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
}

impl Drop for HeadlessRenderer {
    fn drop(&mut self) {
        if self.terminal {
            // Native parents are ManuallyDrop; retain every dependent allocation too.
            std::mem::forget(self.uniforms.take());
            std::mem::forget(self.light.take());
            std::mem::forget(self.instance_buffer.take());
            std::mem::forget(self.depth.take());
            std::mem::forget(self.white_texture.take());
            std::mem::forget(std::mem::take(&mut self.textures));
            std::mem::forget(self.target.take());
            std::mem::forget(self.text.take());
            std::mem::forget(std::mem::take(&mut self.meshes));
            return;
        }
        // Reverse construction order, mirroring VulkanRenderer::Drop: everything allocator-backed
        // before the allocator, the allocator before the device, the device before the instance.
        unsafe {
            if let Some(text) = self.text.take() {
                text.destroy(&self.device, &mut self.allocator);
            }
            ManuallyDrop::drop(&mut self.pipeline);
            if let Some(light) = self.light.take() {
                light.destroy(&self.device, &mut self.allocator);
            }
            if let Some(instance_buffer) = self.instance_buffer.take() {
                instance_buffer.destroy(&self.device, &mut self.allocator);
            }
            for texture in std::mem::take(&mut self.textures) {
                texture.destroy(&self.device, &mut self.allocator);
            }
            if let Some(white_texture) = self.white_texture.take() {
                white_texture.destroy(&self.device, &mut self.allocator);
            }
            self.device
                .device
                .destroy_descriptor_set_layout(self.texture_set_layout, None);
            if let Some(uniforms) = self.uniforms.take() {
                uniforms.destroy(&self.device, &mut self.allocator);
            }
            if let Some(depth) = self.depth.take() {
                depth.destroy(&self.device, &mut self.allocator);
            }
            for mesh in std::mem::take(&mut self.meshes) {
                mesh.destroy(&self.device, &mut self.allocator);
            }
            if let Some(target) = self.target.take() {
                target.destroy(&self.device, &mut self.allocator);
            }
            ManuallyDrop::drop(&mut self.allocator);
            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLOR_ATTACHMENT: vk::FormatFeatureFlags = vk::FormatFeatureFlags::COLOR_ATTACHMENT;

    #[test]
    fn prefers_first_candidate_that_supports_color_attachment() {
        let candidates = [
            (vk::Format::R8G8B8A8_UNORM, COLOR_ATTACHMENT),
            (vk::Format::B8G8R8A8_UNORM, COLOR_ATTACHMENT),
        ];
        assert_eq!(
            select_color_format(&candidates),
            Some(vk::Format::R8G8B8A8_UNORM)
        );
    }

    #[test]
    fn falls_back_when_first_candidate_unsupported() {
        let candidates = [
            (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::empty()),
            (vk::Format::B8G8R8A8_UNORM, COLOR_ATTACHMENT),
        ];
        assert_eq!(
            select_color_format(&candidates),
            Some(vk::Format::B8G8R8A8_UNORM)
        );
    }

    #[test]
    fn returns_none_when_no_candidate_supports_color_attachment() {
        let candidates = [
            (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::empty()),
            (vk::Format::B8G8R8A8_UNORM, vk::FormatFeatureFlags::empty()),
        ];
        assert_eq!(select_color_format(&candidates), None);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn offscreen_target_constructs_tracks_render_state_and_tears_down() {
        let instance = super::super::instance::VulkanInstance::new_headless().expect("instance");
        let device = super::super::device::VulkanDevice::new_headless(&instance).expect("device");
        let mut allocator =
            super::super::allocator::Allocator::new(&instance, &device).expect("allocator");
        let mut target = OffscreenTarget::new(
            &device,
            &instance,
            &mut allocator,
            vk::Extent2D {
                width: 4,
                height: 4,
            },
        )
        .expect("offscreen target");
        assert!(!target.has_rendered());
        assert_eq!(
            target.extent(),
            vk::Extent2D {
                width: 4,
                height: 4,
            }
        );
        assert!(matches!(
            target.format(),
            vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
        ));
        target.mark_rendered();
        assert!(target.has_rendered());
        target.destroy(&device, &mut allocator);
    }

    // Moved from device.rs (Task 1) so every HeadlessRenderer-adjacent GPU test lives together.
    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn headless_instance_and_device_construct_and_tear_down() {
        let instance = super::super::instance::VulkanInstance::new_headless().expect("instance");
        let device = super::super::device::VulkanDevice::new_headless(&instance).expect("device");
        assert_eq!(instance.validation_error_count(), 0);
        drop(device);
        drop(instance);
    }

    /// One outward-normal-per-face unit cube (24 vertices, 36 indices): the same geometry
    /// `examples/cube.rs` builds, inlined here since examples aren't a dependency this crate's tests
    /// can import from. Colors/UVs aren't load-bearing (every `Object` below uses `texture: None`).
    fn lit_untextured_cube() -> (Vec<Vertex>, Vec<Index>) {
        use crate::math::{Vec2, Vec3};

        fn face(normal: [f32; 3], positions: [[f32; 3]; 4]) -> [Vertex; 4] {
            const UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            std::array::from_fn(|i| Vertex {
                position: Vec3::new(positions[i][0], positions[i][1], positions[i][2]),
                color: Vec3::new(1.0, 1.0, 1.0),
                uv: Vec2::new(UVS[i][0], UVS[i][1]),
                normal: Vec3::new(normal[0], normal[1], normal[2]),
            })
        }

        let mut vertices = Vec::with_capacity(24);
        vertices.extend(face(
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ));
        vertices.extend(face(
            [0.0, 0.0, -1.0],
            [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
        ));
        vertices.extend(face(
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ],
        ));
        vertices.extend(face(
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
        ));
        vertices.extend(face(
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
        ));
        vertices.extend(face(
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ],
        ));

        let indices: Vec<Index> = (0..6u16)
            .flat_map(|f| {
                let b = f * 4;
                [b, b + 1, b + 2, b + 2, b + 3, b]
            })
            .collect();

        (vertices, indices)
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn headless_renderer_constructs_and_tears_down() {
        let (vertices, indices) = lit_untextured_cube();
        let renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        drop(renderer);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn render_frame_is_deterministic_across_repeated_calls() {
        use crate::math::{to_radians, Mat4, Vec3};
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let camera = uniform::Camera {
            view: crate::math::look_at(
                &Vec3::new(2.0, 2.0, 2.0),
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: crate::math::perspective(to_radians(45.0), 1.0, 0.1, 10.0),
            position: Vec3::new(2.0, 2.0, 2.0),
        };
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        let first = renderer
            .render_frame(&objects, camera, &DirectionalLight::default(), &[])
            .expect("first render");
        let second = renderer
            .render_frame(&objects, camera, &DirectionalLight::default(), &[])
            .expect("second render");
        assert_eq!(
            first, second,
            "identical scene rendered twice must produce byte-identical output"
        );
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/headless_self_test.png"]
    fn self_test_scene_matches_golden_with_zero_validation_errors() {
        use crate::math::{to_radians, Mat4, Vec3};
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let camera = uniform::Camera {
            view: crate::math::look_at(
                &Vec3::new(2.0, 2.0, 2.0),
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: crate::math::perspective(to_radians(45.0), 1.0, 0.1, 10.0),
            position: Vec3::new(2.0, 2.0, 2.0),
        };
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        let pixels = renderer
            .render_frame(&objects, camera, &DirectionalLight::default(), &[])
            .expect("render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/headless_self_test.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    /// Shared fixed camera for the point-light scenario tests below: same eye/target/up and
    /// projection as `self_test_scene_matches_golden_with_zero_validation_errors`, so every scenario
    /// differs only in its point lights.
    fn point_light_scenario_camera() -> uniform::Camera {
        use crate::math::{to_radians, Vec3};
        uniform::Camera {
            view: crate::math::look_at(
                &Vec3::new(2.0, 2.0, 2.0),
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: crate::math::perspective(to_radians(45.0), 1.0, 0.1, 10.0),
            position: Vec3::new(2.0, 2.0, 2.0),
        }
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/headless_self_test.png (zero-point scenario)"]
    fn point_light_zero_matches_baseline() {
        use crate::math::Mat4;
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        // Zero point lights: proves "zero points changes nothing" by reusing spec 013's existing
        // baseline golden rather than a new one — this scene is otherwise byte-for-byte the same as
        // `self_test_scene_matches_golden_with_zero_validation_errors`.
        let pixels = renderer
            .render_frame(
                &objects,
                point_light_scenario_camera(),
                &DirectionalLight::default(),
                &[],
            )
            .expect("render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/headless_self_test.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/point_light_one_illuminates.png"]
    fn point_light_one_illuminates() {
        use crate::math::{Mat4, Vec3};
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        let point_lights = [PointLight {
            position: Vec3::new(1.5, 0.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            range: 5.0,
        }];
        let pixels = renderer
            .render_frame(
                &objects,
                point_light_scenario_camera(),
                &DirectionalLight::default(),
                &point_lights,
            )
            .expect("render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/point_light_one_illuminates.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/point_light_colors_add.png"]
    fn point_light_colors_add() {
        use crate::math::{Mat4, Vec3};
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        // Two overlapping lights at the same position, pure red and pure blue: if contributions add
        // linearly the lit region should read magenta-ish rather than either pure color.
        let point_lights = [
            PointLight {
                position: Vec3::new(1.2, 0.0, 0.0),
                color: Vec3::new(1.0, 0.0, 0.0),
                range: 5.0,
            },
            PointLight {
                position: Vec3::new(1.2, 0.0, 0.0),
                color: Vec3::new(0.0, 0.0, 1.0),
                range: 5.0,
            },
        ];
        let pixels = renderer
            .render_frame(
                &objects,
                point_light_scenario_camera(),
                &DirectionalLight::default(),
                &point_lights,
            )
            .expect("render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/point_light_colors_add.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/headless_self_test.png (beyond-range scenario)"]
    fn point_light_beyond_range_contributes_zero() {
        use crate::math::{Mat4, Vec3};
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        // Far outside its own range (distance from the unit cube is ~17 units, range is 1): should
        // contribute nothing, so the frame must match the same zero-point baseline golden.
        let point_lights = [PointLight {
            position: Vec3::new(10.0, 10.0, 10.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            range: 1.0,
        }];
        let pixels = renderer
            .render_frame(
                &objects,
                point_light_scenario_camera(),
                &DirectionalLight::default(),
                &point_lights,
            )
            .expect("render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/headless_self_test.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/point_light_eight_lights_and_ambient_does_not_scale.png"]
    fn point_light_eight_lights_and_ambient_does_not_scale() {
        use crate::math::{Mat4, Vec3};
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::light::MAX_POINT_LIGHTS;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        // All MAX_POINT_LIGHTS slots filled, dim and evenly spaced around the cube. If ambient were
        // wrongly multiplied by point-light count, this scene would be visibly blown out relative to
        // the other scenarios' shared ambient floor.
        let point_lights: [PointLight; MAX_POINT_LIGHTS] = std::array::from_fn(|i| {
            let angle = i as f32 * std::f32::consts::TAU / MAX_POINT_LIGHTS as f32;
            PointLight {
                position: Vec3::new(1.2 * angle.cos(), 0.3, 1.2 * angle.sin()),
                color: Vec3::new(0.3, 0.3, 0.3),
                range: 3.0,
            }
        });
        let pixels = renderer
            .render_frame(
                &objects,
                point_light_scenario_camera(),
                &DirectionalLight::default(),
                &point_lights,
            )
            .expect("render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/point_light_eight_lights_and_ambient_does_not_scale.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2).expect("pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    const MULTI_MESH_EXTENT: (u32, u32) = (128, 128);

    fn mesh_scenario_camera() -> uniform::Camera {
        use crate::math::{look_at, perspective, to_radians, Vec3};
        let position = Vec3::new(0.0, 1.0, 4.0);
        uniform::Camera {
            position,
            view: look_at(
                &position,
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: perspective(to_radians(45.0), 1.0, 0.1, 10.0),
        }
    }

    fn mesh_object(mesh: MeshId, x: f32) -> Object {
        Object {
            mesh,
            model: crate::math::translate(&crate::math::Vec3::new(x, 0.0, 0.0)),
            texture: None,
            specular: texture::SpecularMaterial::default(),
        }
    }

    fn triangle_mesh() -> ([Vertex; 3], [Index; 3]) {
        use crate::math::{Vec2, Vec3};
        let vertices = [(-0.45, -0.5), (0.45, -0.5), (0.0, 0.6)].map(|(x, y)| Vertex {
            position: Vec3::new(x, y, 0.0),
            color: Vec3::new(0.2, 0.8, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            uv: Vec2::new(0.0, 0.0),
        });
        (vertices, [0, 1, 2])
    }

    fn assert_mesh_golden(renderer: &HeadlessRenderer, pixels: &[u8], name: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/golden")
            .join(name);
        super::super::golden::assert_matches_golden(
            pixels,
            MULTI_MESH_EXTENT.0,
            MULTI_MESH_EXTENT.1,
            &path,
            2,
        )
        .expect("mesh pixel comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/multi_mesh_different.png"]
    fn multi_mesh_different_geometry_matches_golden() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let (vertices, indices) = lit_untextured_cube();
        let cube = renderer.create_mesh(&vertices, &indices).unwrap();
        let (vertices, indices) = triangle_mesh();
        let triangle = renderer.create_mesh(&vertices, &indices).unwrap();
        let camera = mesh_scenario_camera();
        let light = DirectionalLight::default();
        let pixels = renderer
            .render_frame(
                &[mesh_object(cube, -0.7), mesh_object(triangle, 0.7)],
                camera,
                &light,
                &[],
            )
            .unwrap();
        let wrong_geometry = renderer
            .render_frame(
                &[mesh_object(cube, -0.7), mesh_object(cube, 0.7)],
                camera,
                &light,
                &[],
            )
            .unwrap();
        assert_ne!(
            pixels, wrong_geometry,
            "the second mesh must change actual rendered geometry"
        );
        assert_mesh_golden(&renderer, &pixels, "multi_mesh_different.png");
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/multi_mesh_repeated.png"]
    fn multi_mesh_repeated_handle_matches_golden() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let (vertices, indices) = lit_untextured_cube();
        let mesh = renderer.create_mesh(&vertices, &indices).unwrap();
        let objects = [mesh_object(mesh, -0.7), mesh_object(mesh, 0.7)];
        let camera = mesh_scenario_camera();
        let light = DirectionalLight::default();
        let pixels = renderer
            .render_frame(&objects, camera, &light, &[])
            .unwrap();
        let single = renderer
            .render_frame(&objects[..1], camera, &light, &[])
            .unwrap();
        assert_ne!(
            pixels, single,
            "a repeated handle must still draw the second object"
        );
        assert_mesh_golden(&renderer, &pixels, "multi_mesh_repeated.png");
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/multi_mesh_text_only.png"]
    fn multi_mesh_zero_meshes_text_only_matches_golden() {
        use crate::math::{Vec2, Vec4};
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let camera = mesh_scenario_camera();
        let light = DirectionalLight::default();
        let clear = renderer.render_frame(&[], camera, &light, &[]).unwrap();
        let draws = [TextDraw {
            text: "No meshes\nText works",
            position: Vec2::new(4.0, 4.0),
            scale: 1.0,
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        }];
        let pixels = renderer
            .render_frame_with_text(&[], camera, &light, &[], &draws)
            .unwrap();
        assert!(renderer.meshes.is_empty());
        assert_ne!(
            pixels, clear,
            "text must produce visible pixels with no meshes"
        );
        assert_eq!(
            renderer.render_frame(&[], camera, &light, &[]).unwrap(),
            clear,
            "text must not persist into the next frame"
        );
        assert_mesh_golden(&renderer, &pixels, "multi_mesh_text_only.png");
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; verifies multi-mesh invalid input rejection and reuse"]
    fn multi_mesh_rejects_invalid_handles_and_uploads_then_renders_again() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let (vertices, indices) = lit_untextured_cube();
        let mesh = renderer.create_mesh(&vertices, &indices).unwrap();
        let camera = mesh_scenario_camera();
        let light = DirectionalLight::default();
        let baseline = renderer
            .render_frame(&[mesh_object(mesh, 0.0)], camera, &light, &[])
            .unwrap();
        let count = renderer.meshes.len();
        assert!(renderer.create_mesh(&vertices, &[0, 1, u16::MAX]).is_err());
        assert_eq!(renderer.meshes.len(), count);
        for invalid in [
            MeshId::new(handle::next_owner_id(), 0),
            MeshId::new(renderer.owner, count),
        ] {
            assert!(matches!(
                renderer.render_frame(
                    &[mesh_object(mesh, 0.0), mesh_object(invalid, 0.5)],
                    camera,
                    &light,
                    &[]
                ),
                Err(VulkanError::ValidationError(_))
            ));
        }
        let mut object = mesh_object(mesh, 0.0);
        object.texture = Some(texture::TextureId::new(handle::next_owner_id(), 0));
        assert!(matches!(
            renderer.render_frame(&[object], camera, &light, &[]),
            Err(VulkanError::ValidationError(_))
        ));
        assert_eq!(
            renderer
                .render_frame(&[mesh_object(mesh, 0.0)], camera, &light, &[])
                .unwrap(),
            baseline
        );
        // Simulate terminal state only after completion; no real GPU work is pending here.
        renderer.terminal = true;
        assert!(matches!(
            renderer.render_frame(&[], camera, &light, &[]),
            Err(VulkanError::Terminal(_))
        ));
        assert!(matches!(
            renderer.create_mesh(&vertices, &indices),
            Err(VulkanError::Terminal(_))
        ));
        // Restore the fixture for ordinary teardown; this is not a production recovery path.
        renderer.terminal = false;
        assert_eq!(renderer.validation_error_count(), 0);
    }

    // --- Spec 014 (texture mipmaps): create_texture_with_options ownership/failure-path tests ---

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn create_texture_with_options_rejects_mismatched_pixels_without_publishing_a_texture_id() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let textures_before = renderer.textures.len();
        let sets_before = renderer.texture_sets.len();

        // One byte short of 4x4 RGBA8 (64 bytes): SampledImage's existing validate_pixels check
        // rejects this before any mip-specific GPU work runs, exercising the Generate path's error
        // handling end-to-end without needing a device whose mip capabilities are actually
        // insufficient (impractical to synthesize without real hardware -- see spec 014's
        // "Capability checks and explicit failure policy").
        let result = renderer.create_texture_with_options(
            &[0u8; 63],
            4,
            4,
            TextureOptions {
                mipmaps: texture::Mipmaps::Generate,
            },
        );
        assert!(matches!(result, Err(VulkanError::ValidationError(_))));
        assert_eq!(renderer.textures.len(), textures_before);
        assert_eq!(renderer.texture_sets.len(), sets_before);
        assert!(
            !renderer.terminal,
            "a non-terminal validation error must not leave the renderer terminal"
        );

        // The renderer must still be usable afterward: a correctly-sized request on the same
        // renderer succeeds, proving the failed call above left no half-built state behind.
        assert!(renderer
            .create_texture_with_options(
                &[255u8; 64],
                4,
                4,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Generate,
                },
            )
            .is_ok());
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn successful_generate_call_publishes_exactly_one_texture_id_and_one_descriptor_set() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let textures_before = renderer.textures.len();
        let sets_before = renderer.texture_sets.len();

        let id = renderer
            .create_texture_with_options(
                &[200u8; 64],
                4,
                4,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Generate,
                },
            )
            .expect("a small Generate request must succeed on a real device");

        assert_eq!(renderer.textures.len(), textures_before + 1);
        assert_eq!(renderer.texture_sets.len(), sets_before + 1);

        let white = renderer.white_texture.as_ref().unwrap().set;
        let resolved =
            texture::resolve_texture_set(Some(id), renderer.owner, &renderer.texture_sets, white)
                .expect("the returned TextureId must resolve against this renderer");
        assert_eq!(resolved, *renderer.texture_sets.last().unwrap());
        assert_eq!(renderer.validation_error_count(), 0);
    }

    // --- Spec 014 (texture mipmaps): GPU mip-chain readback and golden-image acceptance tests ---

    /// Synthetic 16x16 RGBA8 readback fixture: four solid, fully-opaque 8x8 quadrants. Per spec
    /// 014's "Resolved decisions", averaging N copies of the same constant value is that same
    /// constant regardless of linear-vs-sRGB blit space, so level 1 (8x8, each texel a 2x2 box
    /// fully inside one quadrant) has an exactly predictable expected value -- no filtering-
    /// tolerance judgment call needed for that level.
    fn quadrant_texture_16x16() -> Vec<u8> {
        const SIZE: usize = 16;
        const RED: [u8; 4] = [255, 0, 0, 255];
        const GREEN: [u8; 4] = [0, 255, 0, 255];
        const BLUE: [u8; 4] = [0, 0, 255, 255];
        const WHITE: [u8; 4] = [255, 255, 255, 255];

        let mut pixels = vec![0u8; SIZE * SIZE * 4];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let color = match (x < SIZE / 2, y < SIZE / 2) {
                    (true, true) => RED,
                    (false, true) => GREEN,
                    (true, false) => BLUE,
                    (false, false) => WHITE,
                };
                let idx = (y * SIZE + x) * 4;
                pixels[idx..idx + 4].copy_from_slice(&color);
            }
        }
        pixels
    }

    /// Procedural checkerboard of `size`x`size` RGBA8 pixels, alternating `a`/`b` in `check_size`-
    /// pixel squares -- high-frequency enough to visibly alias without mipmaps at a far camera
    /// distance (spec 014's demonstration goal), used by the Disabled-vs-Generate golden tests
    /// below. Mirrors the pattern the updated `examples/cube.rs` generates at a larger resolution.
    fn checker_texture(size: u32, check_size: u32, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
        let mut pixels = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let color = if ((x / check_size) + (y / check_size)).is_multiple_of(2) {
                    a
                } else {
                    b
                };
                let idx = ((y * size + x) * 4) as usize;
                pixels[idx..idx + 4].copy_from_slice(&color);
            }
        }
        pixels
    }

    /// Far camera for the checker-textured cube: distant enough that each screen pixel covers many
    /// texels, where Disabled (always sampling level 0) aliases and Generate (sampling a coarser
    /// level) does not.
    fn mip_far_camera() -> uniform::Camera {
        use crate::math::{to_radians, Vec3};
        uniform::Camera {
            view: crate::math::look_at(
                &Vec3::new(8.0, 8.0, 8.0),
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: crate::math::perspective(to_radians(45.0), 1.0, 0.1, 100.0),
            position: Vec3::new(8.0, 8.0, 8.0),
        }
    }

    /// Near camera for the checker-textured cube: close enough that Generate's close-up sampling
    /// (still dominated by level 0) should not look visibly degraded relative to Disabled.
    fn mip_near_camera() -> uniform::Camera {
        use crate::math::{to_radians, Vec3};
        uniform::Camera {
            view: crate::math::look_at(
                &Vec3::new(1.0, 1.0, 1.0),
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: crate::math::perspective(to_radians(45.0), 1.0, 0.1, 10.0),
            position: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn generated_mip_level_1_of_quadrant_texture_matches_each_quadrant_exactly() {
        use crate::renderer::mip_readback::read_back_mip_level;

        let pixels = quadrant_texture_16x16();
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let id = renderer
            .create_texture_with_options(
                &pixels,
                16,
                16,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Generate,
                },
            )
            .expect("16x16 Generate request must succeed on a real device");
        let image = renderer.texture_image(id);

        // Level 1 is 8x8: each texel is an exact 2x2 box fully inside one source quadrant.
        let level1 = read_back_mip_level(&renderer.device, &mut renderer.allocator, image, 1, 8, 8)
            .expect("level 1 readback");
        assert_eq!(level1.len(), 8 * 8 * 4);

        const TOLERANCE: i16 = 2; // golden-image default tolerance, per golden.rs's convention
        let texel_at = |x: usize, y: usize| -> [u8; 4] {
            let idx = (y * 8 + x) * 4;
            [
                level1[idx],
                level1[idx + 1],
                level1[idx + 2],
                level1[idx + 3],
            ]
        };
        let assert_quadrant = |xs: std::ops::Range<usize>,
                               ys: std::ops::Range<usize>,
                               expected: [u8; 4]| {
            for y in ys.clone() {
                for x in xs.clone() {
                    let actual = texel_at(x, y);
                    for ch in 0..4 {
                        let diff = (i16::from(actual[ch]) - i16::from(expected[ch])).abs();
                        assert!(
                            diff <= TOLERANCE,
                            "level-1 texel ({x},{y}) channel {ch}: expected {expected:?}, got {actual:?}"
                        );
                    }
                }
            }
        };
        assert_quadrant(0..4, 0..4, [255, 0, 0, 255]); // top-left: red
        assert_quadrant(4..8, 0..4, [0, 255, 0, 255]); // top-right: green
        assert_quadrant(0..4, 4..8, [0, 0, 255, 255]); // bottom-left: blue
        assert_quadrant(4..8, 4..8, [255, 255, 255, 255]); // bottom-right: white
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver"]
    fn generated_mip_chain_has_expected_level_count_and_shrinking_dimensions() {
        use crate::renderer::mip::mip_extent;
        use crate::renderer::mip_readback::read_back_mip_level;

        let pixels = quadrant_texture_16x16();
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let id = renderer
            .create_texture_with_options(
                &pixels,
                16,
                16,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Generate,
                },
            )
            .expect("16x16 Generate request must succeed on a real device");
        let image = renderer.texture_image(id);

        // 16x16 -> mip_level_count(16, 16) == 5: levels of 16, 8, 4, 2, 1.
        const EXPECTED_LEVEL_COUNT: u32 = 5;
        for level in 0..EXPECTED_LEVEL_COUNT {
            let (w, h) = mip_extent(16, 16, level);
            let bytes = read_back_mip_level(
                &renderer.device,
                &mut renderer.allocator,
                image,
                level,
                w,
                h,
            )
            .unwrap_or_else(|e| panic!("level {level} readback failed: {e:?}"));
            assert_eq!(
                bytes.len(),
                (w * h * 4) as usize,
                "level {level} byte length"
            );
        }
        assert_eq!(renderer.validation_error_count(), 0);
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/mipmap_disabled_far.png and mipmap_generate_far.png"]
    fn mipmap_disabled_vs_generate_differ_at_far_camera_distance() {
        use crate::math::Mat4;
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let checker = checker_texture(64, 4, [220, 40, 40, 255], [255, 255, 255, 255]);

        let mut disabled_renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let disabled_tex = disabled_renderer
            .create_texture_with_options(
                &checker,
                64,
                64,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Disabled,
                },
            )
            .expect("disabled texture upload");
        let disabled_objects = [Object {
            mesh: disabled_renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: Some(disabled_tex),
            specular: SpecularMaterial::default(),
        }];
        let disabled_pixels = disabled_renderer
            .render_frame(
                &disabled_objects,
                mip_far_camera(),
                &DirectionalLight::default(),
                &[],
            )
            .expect("disabled far render");
        assert_eq!(disabled_renderer.validation_error_count(), 0);

        let mut generate_renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let generate_tex = generate_renderer
            .create_texture_with_options(
                &checker,
                64,
                64,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Generate,
                },
            )
            .expect("generate texture upload");
        let generate_objects = [Object {
            mesh: generate_renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: Some(generate_tex),
            specular: SpecularMaterial::default(),
        }];
        let generate_pixels = generate_renderer
            .render_frame(
                &generate_objects,
                mip_far_camera(),
                &DirectionalLight::default(),
                &[],
            )
            .expect("generate far render");
        assert_eq!(generate_renderer.validation_error_count(), 0);

        // Mipmapping must visibly change the far-distance render: the two buffers must not be
        // within (a generous, non-golden) tolerance of each other.
        let differs = disabled_pixels
            .iter()
            .zip(generate_pixels.iter())
            .any(|(a, b)| (i16::from(*a) - i16::from(*b)).abs() > 2);
        assert!(
            differs,
            "Disabled vs Generate renders must differ at a far camera distance"
        );

        let disabled_golden = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/mipmap_disabled_far.png"
        ));
        assert_matches_golden(&disabled_pixels, 64, 64, disabled_golden, 2)
            .expect("disabled far golden comparison");

        let generate_golden = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/mipmap_generate_far.png"
        ));
        assert_matches_golden(&generate_pixels, 64, 64, generate_golden, 2)
            .expect("generate far golden comparison");
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/mipmap_generate_near.png"]
    fn mipmap_generate_at_near_camera_distance_matches_golden() {
        use crate::math::Mat4;
        use crate::renderer::golden::assert_matches_golden;
        use crate::renderer::texture::SpecularMaterial;

        let (vertices, indices) = lit_untextured_cube();
        let checker = checker_texture(64, 4, [220, 40, 40, 255], [255, 255, 255, 255]);

        let mut renderer =
            HeadlessRenderer::new((64, 64), &vertices, &indices).expect("headless renderer");
        let tex = renderer
            .create_texture_with_options(
                &checker,
                64,
                64,
                TextureOptions {
                    mipmaps: texture::Mipmaps::Generate,
                },
            )
            .expect("generate texture upload");
        let objects = [Object {
            mesh: renderer.mesh(),
            model: Mat4::IDENTITY,
            texture: Some(tex),
            specular: SpecularMaterial::default(),
        }];
        let pixels = renderer
            .render_frame(
                &objects,
                mip_near_camera(),
                &DirectionalLight::default(),
                &[],
            )
            .expect("generate near render");

        let golden_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/golden/mipmap_generate_near.png"
        ));
        assert_matches_golden(&pixels, 64, 64, golden_path, 2)
            .expect("generate near golden comparison");
        assert_eq!(renderer.validation_error_count(), 0);
    }

    // --- Spec 016 (GPU instancing): golden-image acceptance tests for the batched instance path ---

    const MANY_INSTANCES_GRID: usize = 5;
    const MANY_INSTANCES_COUNT: usize = MANY_INSTANCES_GRID * MANY_INSTANCES_GRID;
    const MANY_INSTANCES_SPACING: f32 = 0.5;
    /// Row-major index (0-based) of the one instance given a non-uniformly-scaled model matrix --
    /// the grid center, so it's unambiguous in the golden image which cube is stretched.
    const NON_UNIFORM_SCALE_INDEX: usize = 12;

    fn many_instances_camera() -> uniform::Camera {
        use crate::math::{look_at, perspective, to_radians, Vec3};
        let position = Vec3::new(0.0, 3.0, 4.0);
        uniform::Camera {
            position,
            view: look_at(
                &position,
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: perspective(to_radians(60.0), 1.0, 0.1, 20.0),
        }
    }

    /// `MANY_INSTANCES_COUNT` copies of `mesh` on a 5x5 grid, each with an independently different
    /// position, Y rotation, and shininess -- proving a single batched run still carries correct
    /// per-instance data, not a uniform cube merely repeated. One instance (the grid center) gets a
    /// non-uniformly-scaled model matrix (per this plan's Review Focus): a batched draw's
    /// per-instance model matrices must each light correctly on their own, not assume a shared
    /// uniform scale baked into the pipeline.
    fn many_instances_objects(mesh: MeshId) -> Vec<Object> {
        use crate::math::{rotate, to_radians, translate, Mat4, Transform4x4, Vec3};
        (0..MANY_INSTANCES_COUNT)
            .map(|i| {
                let col = (i % MANY_INSTANCES_GRID) as f32;
                let row = (i / MANY_INSTANCES_GRID) as f32;
                let x = (col - 2.0) * MANY_INSTANCES_SPACING;
                let z = (row - 2.0) * MANY_INSTANCES_SPACING;
                let rotation = rotate(&Vec3::new(0.0, 1.0, 0.0), to_radians(i as f32 * 15.0));
                let scale = if i == NON_UNIFORM_SCALE_INDEX {
                    Mat4::scale(Vec3::new(0.7, 0.15, 0.3))
                } else {
                    Mat4::scale(Vec3::new(0.3, 0.3, 0.3))
                };
                Object {
                    mesh,
                    model: translate(&Vec3::new(x, 0.0, z)) * rotation * scale,
                    texture: None,
                    specular: texture::SpecularMaterial {
                        color: Vec3::new(0.8, 0.8, 0.8),
                        shininess: 4.0 + ((i as f32 * 12.0) % 128.0),
                    },
                }
            })
            .collect()
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/instanced_many.png"]
    fn many_instances_one_mesh_single_draw_call_matches_golden() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let (vertices, indices) = lit_untextured_cube();
        let cube = renderer.create_mesh(&vertices, &indices).unwrap();
        let objects = many_instances_objects(cube);
        let camera = many_instances_camera();
        let light = DirectionalLight::default();
        let pixels = renderer
            .render_frame(&objects, camera, &light, &[])
            .unwrap();
        assert_eq!(
            renderer.scene_draw_calls(),
            1,
            "{} instances of one mesh/texture must merge into a single instanced draw call",
            MANY_INSTANCES_COUNT
        );
        assert_eq!(renderer.scene_instances(), MANY_INSTANCES_COUNT);
        assert_mesh_golden(&renderer, &pixels, "instanced_many.png");
    }

    /// Translate + a fixed uniform scale-down, so six objects spread along X don't overlap --
    /// distinct from `mesh_object` (identity scale) above, which only ever places two objects far
    /// enough apart on its own.
    fn mesh_run_object(mesh: MeshId, x: f32) -> Object {
        use crate::math::{translate, Mat4, Transform4x4, Vec3};
        Object {
            mesh,
            model: translate(&Vec3::new(x, 0.0, 0.0)) * Mat4::scale(Vec3::new(0.5, 0.5, 0.5)),
            texture: None,
            specular: texture::SpecularMaterial::default(),
        }
    }

    fn mesh_run_camera() -> uniform::Camera {
        use crate::math::{look_at, perspective, to_radians, Vec3};
        let position = Vec3::new(0.0, 1.2, 5.0);
        uniform::Camera {
            position,
            view: look_at(
                &position,
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(0.0, 1.0, 0.0),
            ),
            proj: perspective(to_radians(50.0), 1.0, 0.1, 20.0),
        }
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/instanced_mixed_runs.png"]
    fn mixed_mesh_runs_three_draw_calls_matches_golden() {
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let (vertices, indices) = lit_untextured_cube();
        let cube = renderer.create_mesh(&vertices, &indices).unwrap();
        let (vertices, indices) = triangle_mesh();
        let triangle = renderer.create_mesh(&vertices, &indices).unwrap();
        // Mesh A x3, mesh B x2, mesh A x1 again: three consecutive runs where the middle one is a
        // different mesh entirely -- proves a run boundary closes and reopens correctly, not just
        // "any two different meshes batch separately".
        let objects = [
            mesh_run_object(cube, -1.5),
            mesh_run_object(cube, -0.9),
            mesh_run_object(cube, -0.3),
            mesh_run_object(triangle, 0.3),
            mesh_run_object(triangle, 0.9),
            mesh_run_object(cube, 1.5),
        ];
        let camera = mesh_run_camera();
        let light = DirectionalLight::default();
        let pixels = renderer
            .render_frame(&objects, camera, &light, &[])
            .unwrap();
        assert_eq!(
            renderer.scene_draw_calls(),
            3,
            "mesh A x3, mesh B x2, mesh A x1 must emit exactly three runs, not fewer or more"
        );
        assert_eq!(renderer.scene_instances(), objects.len());
        assert_mesh_golden(&renderer, &pixels, "instanced_mixed_runs.png");
    }

    #[test]
    #[ignore = "needs a real Vulkan driver; generates/compares testdata/golden/instanced_with_text.png"]
    fn instanced_scene_with_text_matches_golden() {
        use crate::math::{Vec2, Vec4};
        let mut renderer = HeadlessRenderer::new_empty(MULTI_MESH_EXTENT).unwrap();
        let (vertices, indices) = lit_untextured_cube();
        let cube = renderer.create_mesh(&vertices, &indices).unwrap();
        let objects = many_instances_objects(cube);
        let camera = many_instances_camera();
        let light = DirectionalLight::default();
        let draws = [TextDraw {
            text: "instanced",
            position: Vec2::new(4.0, 4.0),
            scale: 1.0,
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        }];
        let pixels = renderer
            .render_frame_with_text(&objects, camera, &light, &[], &draws)
            .unwrap();
        assert_eq!(
            renderer.scene_draw_calls(),
            1,
            "the text pass must not change the underlying instanced scene's draw-call count"
        );
        assert_eq!(renderer.scene_instances(), MANY_INSTANCES_COUNT);
        let without_text = renderer
            .render_frame(&objects, camera, &light, &[])
            .unwrap();
        assert_ne!(
            pixels, without_text,
            "text must produce visible pixels over the instanced scene"
        );
        assert_mesh_golden(&renderer, &pixels, "instanced_with_text.png");
    }
}
