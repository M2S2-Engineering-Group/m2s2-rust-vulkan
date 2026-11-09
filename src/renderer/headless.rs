// Offscreen color render target for headless (no-surface) rendering in GPU tests. Stands in for a
// swapchain image: no surface, no present — just a plain color `VkImage` the one-shot recorder
// (Task 3) renders into and reads back from. See specs/019-headless-rendering-and-golden-image-tests.md.

use crate::error::{Result, VulkanError};
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::{Buffer, Index, Vertex};
use crate::renderer::command::{self, ObjectCommand};
use crate::renderer::depth::DepthResources;
use crate::renderer::device::VulkanDevice;
use crate::renderer::instance::VulkanInstance;
use crate::renderer::light::{DirectionalLight, Light};
use crate::renderer::pipeline::{self, GraphicsPipeline};
use crate::renderer::submission::{SubmitState, SubmitStep};
use crate::renderer::texture::{self, Object, StoredTexture};
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
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    depth: Option<DepthResources>,
    uniforms: Option<Uniforms>,
    texture_set_layout: vk::DescriptorSetLayout,
    white_texture: Option<StoredTexture>,
    light: Option<Light>,
    pipeline: ManuallyDrop<GraphicsPipeline>,
    target: Option<OffscreenTarget>,
}

impl HeadlessRenderer {
    /// Mirrors `VulkanRenderer::init`'s resource-construction order (spec 012's `Option`-held-locals
    /// rollback-on-error pattern), substituting the headless instance/device and an `OffscreenTarget`
    /// for the swapchain. `extent` sizes the depth attachment and the offscreen color target alike.
    pub(crate) fn new(extent: (u32, u32), vertices: &[Vertex], indices: &[Index]) -> Result<Self> {
        let instance = VulkanInstance::new_headless()?;
        let device = VulkanDevice::new_headless(&instance)?;
        let mut allocator = Allocator::new(&instance, &device)?;

        let extent_2d = vk::Extent2D {
            width: extent.0,
            height: extent.1,
        };

        let mut vertex_buffer = None;
        let mut index_buffer = None;
        let mut depth = None;
        let mut uniforms = None;
        let mut texture_set_layout = None;
        let mut white_texture = None;
        let mut light = None;
        let mut target = None;
        let resources = (|| {
            vertex_buffer = Some(Buffer::new_device_local(
                &device,
                &mut allocator,
                vertices,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?);
            index_buffer = Some(Buffer::new_device_local(
                &device,
                &mut allocator,
                indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?);

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
                layout,
                &[255, 255, 255, 255],
                1,
                1,
            )?);

            light = Some(Light::new(&device, &mut allocator)?);

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
                if let Some(resource) = index_buffer.take() {
                    resource.destroy(&device, &mut allocator);
                }
                if let Some(resource) = vertex_buffer.take() {
                    resource.destroy(&device, &mut allocator);
                }
                return Err(error);
            }
        };

        Ok(Self {
            device: ManuallyDrop::new(device),
            instance: ManuallyDrop::new(instance),
            allocator: ManuallyDrop::new(allocator),
            vertex_buffer,
            index_buffer,
            depth,
            uniforms,
            texture_set_layout: texture_set_layout.unwrap(),
            white_texture,
            light,
            pipeline: ManuallyDrop::new(pipeline),
            target,
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
    ) -> Result<Vec<u8>> {
        self.uniforms
            .as_mut()
            .expect("uniforms present outside teardown")
            .update(0, &uniform::CameraUniforms::from(&camera))?;
        self.light
            .as_mut()
            .expect("light present outside teardown")
            .update(0, &DirectionalLight::default())?;

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

            let vertex_buffer = self
                .vertex_buffer
                .as_ref()
                .expect("vertex buffer present outside teardown");
            let index_buffer = self
                .index_buffer
                .as_ref()
                .expect("index buffer present outside teardown");
            dev.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer.buffer], &[0]);
            dev.cmd_bind_index_buffer(cmd, index_buffer.buffer, 0, vk::IndexType::UINT16);

            let white_set = self
                .white_texture
                .as_ref()
                .expect("white texture present outside teardown")
                .set;
            command::emit_object_draws(objects, index_buffer.len(), &[], white_set, |command| {
                match command {
                    ObjectCommand::BindTexture(set) => dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        1,
                        &[set],
                        &[],
                    ),
                    ObjectCommand::PushModel(bytes) => dev.cmd_push_constants(
                        cmd,
                        pipeline_layout,
                        pipeline::MODEL_PUSH_STAGES,
                        pipeline::MODEL_PUSH_OFFSET,
                        bytes,
                    ),
                    ObjectCommand::DrawIndexed(count) => {
                        dev.cmd_draw_indexed(cmd, count, 1, 0, 0, 0);
                    }
                }
            });

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
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }

        self.target
            .as_mut()
            .expect("target present outside teardown")
            .mark_rendered();

        Ok(pixels)
    }

    /// Forwards to the underlying headless instance's validation-error counter (see
    /// `VulkanInstance::validation_error_count`), so tests can assert "zero validation errors"
    /// without reaching into private fields.
    pub(crate) fn validation_error_count(&self) -> u32 {
        self.instance.validation_error_count()
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
        // Reverse construction order, mirroring VulkanRenderer::Drop: everything allocator-backed
        // before the allocator, the allocator before the device, the device before the instance.
        unsafe {
            ManuallyDrop::drop(&mut self.pipeline);
            if let Some(light) = self.light.take() {
                light.destroy(&self.device, &mut self.allocator);
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
            if let Some(index_buffer) = self.index_buffer.take() {
                index_buffer.destroy(&self.device, &mut self.allocator);
            }
            if let Some(vertex_buffer) = self.vertex_buffer.take() {
                vertex_buffer.destroy(&self.device, &mut self.allocator);
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
            model: Mat4::IDENTITY,
            texture: None,
            specular: SpecularMaterial::default(),
        }];
        let first = renderer
            .render_frame(&objects, camera)
            .expect("first render");
        let second = renderer
            .render_frame(&objects, camera)
            .expect("second render");
        assert_eq!(
            first, second,
            "identical scene rendered twice must produce byte-identical output"
        );
        assert_eq!(renderer.validation_error_count(), 0);
    }
}
