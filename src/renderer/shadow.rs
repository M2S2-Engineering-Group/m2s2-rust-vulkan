//! Caller-facing directional shadow configuration and the renderer-owned per-frame GPU resources
//! that back it: descriptor set 3 (binding 0 combined depth-comparison sampler, FRAGMENT; binding 1
//! light-view-projection UBO, VERTEX | FRAGMENT). See specs/017-directional-shadow-mapping.md,
//! "Resolved decisions". Both rendering paths share configuration validation and transactional
//! pipeline/resource replacement here.

use crate::error::{Result, VulkanError};
use crate::math::Mat4;
use crate::renderer::allocator::Allocator;
use crate::renderer::buffer::Buffer;
use crate::renderer::command::FRAMES_IN_FLIGHT;
use crate::renderer::device::VulkanDevice;
use crate::renderer::instance::VulkanInstance;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// A directional light's shadow-map configuration (spec 017's "Public API" section). At the
/// `VulkanRenderer` level `None` means no shadow pass; `Some(DirectionalShadow)` carries the
/// caller-supplied light view-projection matrix and raster depth-bias factors the depth-only shadow
/// pass uses. The library does not derive `light_view_proj` from a direction -- the caller supplies
/// it already aimed consistently with the configured sun, using the renderer's Vulkan clip
/// conventions (depth `[0,1]`, exactly one Y adjustment -- see `crate::math::perspective`).
#[derive(Copy, Clone, Debug)]
pub struct DirectionalShadow {
    pub light_view_proj: Mat4,
    /// Square shadow-map resolution in texels. Must be nonzero.
    pub resolution: u32,
    /// Raster depth-bias constant factor. Any finite signed value is accepted.
    pub constant_bias: f32,
    /// Raster depth-bias slope factor. Any finite signed value is accepted.
    pub slope_bias: f32,
}

/// Rejects `resolution == 0`, any non-finite element of `light_view_proj` (all 16 components), or a
/// non-finite `constant_bias`/`slope_bias`. Bias may otherwise be any signed finite value. Pure, so
/// it is unit-testable without a device -- mirrors `light.rs`'s `DirectionalLight::set_validated` and
/// `texture.rs`'s `validate_specular`.
pub(crate) fn validate_directional_shadow(shadow: &DirectionalShadow) -> Result<()> {
    if shadow.resolution == 0 {
        return Err(VulkanError::ValidationError(
            "directional shadow resolution must be nonzero".into(),
        ));
    }
    let matrix_finite = (0..4)
        .flat_map(|row| (0..4).map(move |col| shadow.light_view_proj[row][col]))
        .all(f32::is_finite);
    if !matrix_finite {
        return Err(VulkanError::ValidationError(
            "directional shadow light_view_proj must be finite".into(),
        ));
    }
    if !shadow.constant_bias.is_finite() || !shadow.slope_bias.is_finite() {
        return Err(VulkanError::ValidationError(
            "directional shadow constant_bias and slope_bias must be finite".into(),
        ));
    }
    Ok(())
}

// Shadow depth formats in preference order: plain 32-bit float depth first, then the 16-bit
// fallback. Both are stencil-free (the shadow pass never uses a stencil aspect -- see
// `select_shadow_depth_format`'s DEPTH_STENCIL_ATTACHMENT + SAMPLED_IMAGE requirement).
pub(crate) const SHADOW_DEPTH_FORMAT_CANDIDATES: [vk::Format; 2] =
    [vk::Format::D32_SFLOAT, vk::Format::D16_UNORM];

/// Picks the shadow depth format from `candidates` (priority order, paired with each format's
/// queried `optimal_tiling_features`). Unlike `depth.rs`'s `select_depth_format`, this requires BOTH
/// `DEPTH_STENCIL_ATTACHMENT` (usable as a depth attachment) AND `SAMPLED_IMAGE` (sampled by the main
/// pass's comparison sampler) -- not `SAMPLED_IMAGE_FILTER_LINEAR`, since the comparison sampler uses
/// `NEAREST`, no linear-filtering requirement. Returns the first format whose features advertise
/// both, or `None` if none qualify. Pure given the queried properties, so it's testable without a
/// device.
pub(crate) fn select_shadow_depth_format(
    candidates: &[(vk::Format, vk::FormatFeatureFlags)],
) -> Option<vk::Format> {
    let required =
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE;
    candidates
        .iter()
        .find(|(_, features)| features.contains(required))
        .map(|&(format, _)| format)
}

/// The binding index of the combined depth-comparison sampler within set 3 (FRAGMENT only -- only
/// the main pass samples it).
const SHADOW_SAMPLER_BINDING: u32 = 0;
/// The binding index of the light-view-projection UBO within set 3 (VERTEX | FRAGMENT -- the main
/// pass's vertex shader needs it to compute each vertex's light-space position, and the shadow
/// pass's own vertex shader needs it to transform vertices into light clip space).
const SHADOW_LIGHT_VP_BINDING: u32 = 1;

/// One frame-in-flight's shadow resources: a depth image/view/allocation sized `resolution ×
/// resolution`, a host-visible mapped `Buffer` carrying that frame's light-view-projection matrix (64
/// bytes), and the descriptor set binding both into set 3. Not `Drop`: freeing the allocation needs
/// `&mut Allocator`, so `ShadowResources::destroy` calls `ShadowSlot::destroy` explicitly.
struct ShadowSlot {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    buffer: Buffer,
    descriptor_set: vk::DescriptorSet,
}

impl ShadowSlot {
    /// Destroy the view + image (via `device`), free the allocation (via `allocator`), and destroy
    /// the light-VP buffer. The descriptor set itself is freed implicitly when the shared descriptor
    /// pool is destroyed (`ShadowResources::destroy`), same as `PerFrameUniform::destroy`.
    fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        unsafe {
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation {
            let _ = allocator.free(allocation);
        }
        self.buffer.destroy(device, allocator);
    }
}

fn shadow_image_usage() -> vk::ImageUsageFlags {
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    // Only GPU tests copy the depth map to host memory.
    if cfg!(test) {
        usage | vk::ImageUsageFlags::TRANSFER_SRC
    } else {
        usage
    }
}

/// Builds one `ShadowSlot`: depth image -> allocation -> bind -> view -> light-VP buffer ->
/// descriptor-set allocation + write, rolling back every step already completed if a later one
/// fails (mirrors `depth.rs:67-127`'s image/allocation/view rollback, extended with `texture.rs`'s
/// `StoredTexture::new` pattern for the trailing descriptor-set step).
#[allow(clippy::too_many_arguments)]
fn build_shadow_slot(
    device: &VulkanDevice,
    allocator: &mut Allocator,
    format: vk::Format,
    resolution: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
) -> Result<ShadowSlot> {
    let dev = &device.device;

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: resolution,
            height: resolution,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(shadow_image_usage())
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe {
        dev.create_image(&image_info, None)
            .map_err(VulkanError::from)?
    };

    let requirements = unsafe { dev.get_image_memory_requirements(image) };
    let allocation_result = allocator.allocate(&AllocationCreateDesc {
        name: "shadow depth",
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false, // OPTIMAL-tiled image
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    });
    let allocation = match allocation_result {
        Ok(allocation) => allocation,
        Err(error) => {
            unsafe { dev.destroy_image(image, None) };
            return Err(error);
        }
    };

    let view_result = (|| {
        unsafe {
            dev.bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(VulkanError::from)?;
        }
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            dev.create_image_view(&view_info, None)
                .map_err(VulkanError::from)
        }
    })();
    let view = match view_result {
        Ok(view) => view,
        Err(error) => {
            unsafe { dev.destroy_image(image, None) };
            let _ = allocator.free(allocation);
            return Err(error);
        }
    };

    let buffer_result = Buffer::new_host_visible_mapped(
        device,
        allocator,
        std::mem::size_of::<Mat4>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    );
    let buffer = match buffer_result {
        Ok(buffer) => buffer,
        Err(error) => {
            unsafe {
                dev.destroy_image_view(view, None);
                dev.destroy_image(image, None);
            }
            let _ = allocator.free(allocation);
            return Err(error);
        }
    };

    let set_result = (|| {
        let set = unsafe {
            dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_set_layout]),
            )
        }
        .map_err(VulkanError::from)?[0];
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .sampler(sampler)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(std::mem::size_of::<Mat4>() as u64)];
        unsafe {
            dev.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(SHADOW_SAMPLER_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(SHADOW_LIGHT_VP_BINDING)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_info),
                ],
                &[],
            );
        }
        Ok(set)
    })();
    match set_result {
        Ok(descriptor_set) => Ok(ShadowSlot {
            image,
            view,
            allocation: Some(allocation),
            buffer,
            descriptor_set,
        }),
        Err(error) => {
            buffer.destroy(device, allocator);
            unsafe {
                dev.destroy_image_view(view, None);
                dev.destroy_image(image, None);
            }
            let _ = allocator.free(allocation);
            Err(error)
        }
    }
}

/// Owns, per frame-in-flight (`[_; FRAMES_IN_FLIGHT]`): a shadow depth image + view + allocation at
/// `resolution × resolution`, a light-view-proj host-visible mapped buffer, and a descriptor set
/// (set-3 shape: binding 0 = that slot's image view + the one shared comparison sampler, binding 1 =
/// that slot's light-VP buffer). Shared across slots: one descriptor-set layout, one descriptor pool
/// (sized for `FRAMES_IN_FLIGHT` sets of each descriptor type), one comparison `vk::Sampler`, and a
/// `[bool; FRAMES_IN_FLIGHT]` "has this slot ever been rendered into" tracker (spec 017's "First-use
/// vs. reuse layout tracking"; all `false` at construction). Not `Drop`: freeing the allocations
/// needs `&mut Allocator`, so call `destroy` explicitly.
pub(crate) struct ShadowResources {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    format: vk::Format,
    slots: [ShadowSlot; FRAMES_IN_FLIGHT],
    used: [bool; FRAMES_IN_FLIGHT],
}

impl ShadowResources {
    /// Queries `get_physical_device_format_properties` for each of
    /// `SHADOW_DEPTH_FORMAT_CANDIDATES`, picks a format via `select_shadow_depth_format`, fails with
    /// a descriptive `VulkanError::InitializationError` if neither candidate qualifies, then builds
    /// the shared sampler/layout/pool and one `ShadowSlot` per frame-in-flight. Rolls back every
    /// partially-constructed resource on any later failure (mirrors `StoredTexture::new`'s rollback
    /// shape): a failure building slot N destroys slots `0..N` plus the shared pool/layout/sampler;
    /// a failure building the pool destroys the layout + sampler; a failure building the layout
    /// destroys the sampler.
    pub(crate) fn new(
        device: &VulkanDevice,
        instance: &VulkanInstance,
        allocator: &mut Allocator,
        resolution: u32,
    ) -> Result<Self> {
        let candidates: Vec<(vk::Format, vk::FormatFeatureFlags)> = SHADOW_DEPTH_FORMAT_CANDIDATES
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
        let format = select_shadow_depth_format(&candidates).ok_or_else(|| {
            VulkanError::InitializationError(
                "no supported shadow depth format with DEPTH_STENCIL_ATTACHMENT and SAMPLED_IMAGE"
                    .into(),
            )
        })?;

        let properties = unsafe {
            instance
                .instance
                .get_physical_device_image_format_properties(
                    device.physical_device,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    shadow_image_usage(),
                    vk::ImageCreateFlags::empty(),
                )
        }
        .map_err(VulkanError::from)?;
        if resolution > properties.max_extent.width
            || resolution > properties.max_extent.height
            || !properties
                .sample_counts
                .contains(vk::SampleCountFlags::TYPE_1)
        {
            return Err(VulkanError::ValidationError(
                "unsupported directional shadow image extent or sample count".into(),
            ));
        }

        let dev = &device.device;

        // Nearest comparison sampler: no linear filtering (matches `select_shadow_depth_format`'s
        // SAMPLED_IMAGE-only requirement), LESS_OR_EQUAL comparison, clamp-to-edge, no anisotropy,
        // single mip level.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .compare_enable(true)
            .compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .min_lod(0.0)
            .max_lod(0.0);
        let sampler = unsafe {
            dev.create_sampler(&sampler_info, None)
                .map_err(VulkanError::from)?
        };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(SHADOW_SAMPLER_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(SHADOW_LIGHT_VP_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        ];
        let descriptor_set_layout = match unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { dev.destroy_sampler(sampler, None) };
                return Err(VulkanError::from(error));
            }
        };

        // Pool sized for exactly one set of each descriptor type per frame-in-flight.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
        ];
        let descriptor_pool = match unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    dev.destroy_sampler(sampler, None);
                }
                return Err(VulkanError::from(error));
            }
        };

        let mut slots_vec: Vec<ShadowSlot> = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            match build_shadow_slot(
                device,
                allocator,
                format,
                resolution,
                descriptor_pool,
                descriptor_set_layout,
                sampler,
            ) {
                Ok(slot) => slots_vec.push(slot),
                Err(error) => {
                    for slot in slots_vec {
                        slot.destroy(device, allocator);
                    }
                    unsafe {
                        dev.destroy_descriptor_pool(descriptor_pool, None);
                        dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                        dev.destroy_sampler(sampler, None);
                    }
                    return Err(error);
                }
            }
        }
        let slots: [ShadowSlot; FRAMES_IN_FLIGHT] = slots_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("pushed exactly FRAMES_IN_FLIGHT slots"));

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            sampler,
            format,
            slots,
            used: [false; FRAMES_IN_FLIGHT],
        })
    }

    pub(crate) fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub(crate) fn descriptor_set(&self, frame: usize) -> vk::DescriptorSet {
        self.slots[frame].descriptor_set
    }

    pub(crate) fn image(&self, frame: usize) -> vk::Image {
        self.slots[frame].image
    }

    pub(crate) fn view(&self, frame: usize) -> vk::ImageView {
        self.slots[frame].view
    }

    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }

    /// Write `light_view_proj` into frame `frame`'s mapped buffer. The caller must have waited on
    /// that frame's in-flight fence first (the render loop does), so the GPU isn't mid-read. Bias is
    /// NOT stored here -- it's dynamic state at record time (spec 017's "Configuration timing").
    pub(crate) fn update(&mut self, frame: usize, light_view_proj: Mat4) -> Result<()> {
        self.slots[frame]
            .buffer
            .write_mapped(bytemuck::bytes_of(&light_view_proj))
    }

    /// Mark frame `frame`'s shadow depth image as having completed at least one shadow-pass
    /// recording -- Task 5's barrier logic reads this to decide whether the next shadow-pass use
    /// transitions from `UNDEFINED` (first use) or `DEPTH_STENCIL_READ_ONLY_OPTIMAL` (reuse after the
    /// main pass sampled it).
    pub(crate) fn mark_used(&mut self, frame: usize) {
        self.used[frame] = true;
    }

    pub(crate) fn was_used(&self, frame: usize) -> bool {
        self.used[frame]
    }

    /// Destroy every slot's image/view/allocation/buffer, then the shared descriptor pool (which
    /// implicitly frees its descriptor sets), descriptor set layout, and sampler. Must run before the
    /// `Allocator` is dropped.
    pub(crate) fn destroy(self, device: &VulkanDevice, allocator: &mut Allocator) {
        for slot in self.slots {
            slot.destroy(device, allocator);
        }
        unsafe {
            device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.device.destroy_sampler(self.sampler, None);
        }
    }
}

/// Build a complete replacement before either renderer retires its live resources.
/// On failure, every newly allocated resource is rolled back and the old owners are untouched.
pub(super) struct ShadowReplacement {
    pub resources: Option<ShadowResources>,
    pub depth_pipeline: Option<super::pipeline::GraphicsPipeline>,
    pub scene_pipeline: super::pipeline::GraphicsPipeline,
}

pub(super) fn scene_pipeline(
    device: &VulkanDevice,
    formats: (vk::Format, vk::Format),
    layouts: [vk::DescriptorSetLayout; 3],
    shadow: Option<&ShadowResources>,
) -> Result<super::pipeline::GraphicsPipeline> {
    use super::pipeline::GraphicsPipeline;
    match shadow {
        Some(shadow) => GraphicsPipeline::new_with_shadow(
            device,
            formats.0,
            formats.1,
            &[
                layouts[0],
                layouts[1],
                layouts[2],
                shadow.descriptor_set_layout(),
            ],
        ),
        None => GraphicsPipeline::new(device, formats.0, formats.1, &layouts),
    }
}

impl ShadowReplacement {
    pub(super) fn new(
        device: &VulkanDevice,
        instance: &VulkanInstance,
        allocator: &mut Allocator,
        resolution: Option<u32>,
        formats: (vk::Format, vk::Format),
        layouts: [vk::DescriptorSetLayout; 3],
    ) -> Result<Self> {
        let Some(resolution) = resolution else {
            return Ok(Self {
                resources: None,
                depth_pipeline: None,
                scene_pipeline: scene_pipeline(device, formats, layouts, None)?,
            });
        };
        let resources = ShadowResources::new(device, instance, allocator, resolution)?;
        let depth_pipeline = match super::pipeline::GraphicsPipeline::new_shadow_depth(
            device,
            resources.format(),
            &[resources.descriptor_set_layout()],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                resources.destroy(device, allocator);
                return Err(error);
            }
        };
        match scene_pipeline(device, formats, layouts, Some(&resources)) {
            Ok(scene_pipeline) => Ok(Self {
                resources: Some(resources),
                depth_pipeline: Some(depth_pipeline),
                scene_pipeline,
            }),
            Err(error) => {
                drop(depth_pipeline);
                resources.destroy(device, allocator);
                Err(error)
            }
        }
    }
}

pub(super) fn validate_device_shadow(
    config: &DirectionalShadow,
    instance: &VulkanInstance,
    device: &VulkanDevice,
) -> Result<()> {
    validate_directional_shadow(config)?;
    let max_dimension = unsafe {
        instance
            .instance
            .get_physical_device_properties(device.physical_device)
    }
    .limits
    .max_image_dimension2_d;
    if config.resolution > max_dimension {
        return Err(VulkanError::ValidationError(format!(
            "directional shadow resolution {} exceeds device max_image_dimension2D {}",
            config.resolution, max_dimension
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_shadow() -> DirectionalShadow {
        DirectionalShadow {
            light_view_proj: Mat4::IDENTITY,
            resolution: 1024,
            constant_bias: 0.25,
            slope_bias: 1.5,
        }
    }

    #[test]
    fn mat4_payload_is_64_bytes() {
        // ShadowSlot's light-VP buffer and the set-3 UBO binding both size themselves off this --
        // spec 017 fixes the payload at one row-major mat4, 64 bytes.
        assert_eq!(std::mem::size_of::<Mat4>(), 64);
    }

    #[test]
    fn validate_directional_shadow_rejects_zero_resolution() {
        let mut s = valid_shadow();
        s.resolution = 0;
        assert!(validate_directional_shadow(&s).is_err());
    }

    #[test]
    fn validate_directional_shadow_rejects_non_finite_matrix_or_bias() {
        let base = valid_shadow();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut s = base;
            s.light_view_proj[0][0] = bad; // any one element suffices
            assert!(validate_directional_shadow(&s).is_err());
            let mut s = base;
            s.constant_bias = bad;
            assert!(validate_directional_shadow(&s).is_err());
            let mut s = base;
            s.slope_bias = bad;
            assert!(validate_directional_shadow(&s).is_err());
        }
    }

    #[test]
    fn validate_directional_shadow_accepts_signed_finite_bias() {
        let mut s = valid_shadow();
        s.constant_bias = -1.5;
        s.slope_bias = -0.5;
        assert!(validate_directional_shadow(&s).is_ok());
    }

    #[test]
    fn select_shadow_depth_format_prefers_d32_requires_both_features() {
        let both = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        let candidates = [
            (vk::Format::D32_SFLOAT, both),
            (vk::Format::D16_UNORM, both),
        ];
        assert_eq!(
            select_shadow_depth_format(&candidates),
            Some(vk::Format::D32_SFLOAT)
        );
    }

    #[test]
    fn select_shadow_depth_format_falls_back_when_first_lacks_sampled_image() {
        let both = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        let attachment_only = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        let candidates = [
            (vk::Format::D32_SFLOAT, attachment_only),
            (vk::Format::D16_UNORM, both),
        ];
        assert_eq!(
            select_shadow_depth_format(&candidates),
            Some(vk::Format::D16_UNORM)
        );
    }

    #[test]
    fn select_shadow_depth_format_returns_none_when_no_candidate_qualifies() {
        let candidates = [
            (
                vk::Format::D32_SFLOAT,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
            (vk::Format::D16_UNORM, vk::FormatFeatureFlags::SAMPLED_IMAGE),
        ];
        assert_eq!(select_shadow_depth_format(&candidates), None);
    }
}
