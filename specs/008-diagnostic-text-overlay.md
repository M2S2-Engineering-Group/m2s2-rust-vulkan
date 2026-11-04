# 008: Text drawing and diagnostic overlay

**Status**: Implemented.
**Roadmap item**: Phase 2, diagnostic text overlay.

## Goal

Provide reusable screen-space text rendering, then display camera position, CPU frame cadence, and
Vulkan state using it. Diagnostics formatting has no GPU ownership; text rendering has no diagnostic
knowledge. The caller retains window, input, camera, and event-loop ownership.

## Non-goals

General GUI controls, text wrapping/shaping/localization, world-space labels, GPU timestamps, memory
budgets, pipeline statistics, and validation history. CPU frame interval is never labeled GPU time.

## Public API

Re-export from the crate root:

```rust
pub struct TextDraw<'a> {
    pub text: &'a str,
    pub position: Vec2,
    pub scale: f32,
    pub color: Vec4,
}

impl VulkanRenderer {
    pub fn render_objects_with_text(
        &mut self, models: &[Mat4], view: Mat4, proj: Mat4, text: &[TextDraw<'_>],
    ) -> Result<()>;
    pub fn stats(&self) -> RendererStats<'_>;
}

pub enum RenderStatus { Ready, ResizePending, Paused }

pub struct RendererStats<'a> {
    pub device_name: &'a str,
    pub extent: (u32, u32),
    pub color_format: vk::Format,
    pub image_count: usize,
    pub scene_draw_calls: usize,
    pub submitted_indices: u64,
    pub status: RenderStatus,
}

// Default constructs an empty timing window.
pub struct Diagnostics { /* private CPU timing samples */ }
impl Diagnostics {
    pub fn update(&mut self, frame_interval: Duration);
    pub fn reset(&mut self);
    pub fn timing(&self) -> Option<(f64, f64)>; // FPS, milliseconds
    pub fn format(&self, stats: &RendererStats<'_>, camera_position: Vec3) -> String;
}
```

Existing `render_frame`/`render_objects` forward to the same frame path with empty text. Text is
borrowed only for that call; no persistent print queue or hidden accumulation. Empty scene models
still allow text over the clear color. Empty text emits no overlay commands.

Position is measured in physical pixels from the top-left. Scale multiplies the native 8-pixel glyph;
newline advances 10 pixels times scale. Color is linear RGBA in [0,1]. The example uses the window's
scale factor for legibility and maps F3 to visibility.

Unsupported characters become `?`; spaces advance without geometry. Scissoring clips partially visible
glyphs and fully offscreen glyphs are skipped. Invalid scale/position/color or more than 2048 visible
glyphs returns a validation error before acquiring/submitting a frame. Minimized zero-extent frames
skip rendering. No wrapping or silent capacity truncation.

`Diagnostics` averages the last 60 positive caller-provided frame intervals; zero samples are ignored.
FPS is reciprocal average CPU interval, including any application waits, not monitor presentation rate.
Before samples exist, timing is N/A. Reset clears the window. Camera position is supplied explicitly.

Device name is cached at renderer initialization. Snapshot counts describe the last submitted scene
frame, excluding overlay draws. They stay unchanged on skipped acquisition; zero objects submitted
sets them to zero. `FrameOutcome::SwapchainOutOfDate` now carries `submitted: bool` to distinguish
out-of-date acquisition from a submission followed by out-of-date/suboptimal presentation.

## Vulkan objects and ownership

`TextRenderer` lazily owns a sampled R8 font image/allocation, view, nearest sampler, descriptor layout,
pool/set, alpha-blended pipeline, and one fixed-capacity mapped vertex buffer per frame slot. Six
vertices per glyph avoid a separate index buffer. CPU geometry storage is reused between frames.

`image.rs` provides the shared sampled-R8 upload path. `upload.rs` extracts the one-time submission
already used by buffers; image and buffer uploads use it. Future texture formats should extend this
path, not add another uploader. `GraphicsPipeline::create` shares construction between scene/text
pipelines while declaring their different vertex inputs, depth, blending, and push-constant ranges.

Image upload transitions UNDEFINED -> TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL, with
transfer-write -> fragment-read visibility. Startup upload waits on a fence. Per-frame mapped text
updates wait for their frame slot's fence; there is no overlay-specific per-frame wait-idle.

Text records after scene geometry inside the existing dynamic-rendering scope. The pipeline uses the
same color/depth attachment formats, disables depth testing/writing and culling, and alpha blends
font coverage with the supplied color. It reuses the explicitly set viewport/scissor. Rebuild pipeline
objects when attachment formats change; extent-only resize needs dynamic viewport/scissor updates.

All constructor failures release completed resources. Shutdown waits for idle, destroys frame commands,
text pipeline and mapped buffers, descriptor pool/layout, sampler/view/image and allocation, then the
remaining scene resources, allocator, swapchain, device, and instance. Text buffers and images never
outlive the allocator/device.

## Font provenance and build

Public-domain `font8x8_basic.h` from [Daniel Hepper's font8x8](https://github.com/dhepper/font8x8/blob/master/font8x8_basic.h)
is bundled with its original notice. `assets/README.md` documents provenance and
`python3 scripts/generate-font.py` regenerates the packed glyph bytes. The atlas is expanded at startup.
No new dependencies or runtime downloads are needed.

Compile shader changes offline and commit the SPIR-V alongside source:

```sh
glslc shaders/text.vert -o shaders/text.vert.spv
glslc shaders/text.frag -o shaders/text.frag.spv
spirv-val shaders/text.vert.spv
spirv-val shaders/text.frag.spv
```

## Verification and outcome

CPU tests cover atlas bits, vertex layout and shader parsing, positioning/scaling/color, separate text
draws, newline/fallback characters, empty/new-frame reset, clipping/zero extent, invalid inputs and
capacity boundary/overflow; timing's positive/zero/reset/window cases; explicit camera and metric
formatting; sampled-image byte validation; and acquire-versus-submit outcomes. All 48 tests pass, as do
formatting, Clippy with warnings denied, all-target build, and diff whitespace checks.

Ran the cube with synchronization validation enabled. A temporary automated probe exercised resize,
hide/show, minimize/restore, clear-only text, single-object text, scene count assertions, and normal
shutdown; it exited successfully without emitted validation errors. Captured a Vulkan frame with the
SDK screenshot layer and visually verified readable camera, FPS/CPU frame, GPU, swapchain, scene
counts, and state text over the cubes. The temporary probe was removed.

The first draft proposed a diagnostics-specific persistent API. Following design discussion, the final
implementation uses per-frame text submissions and separate CPU diagnostics instead. General text
rendering is available without constructing `Diagnostics`. GPU timestamps remain deferred. Testing on
a second platform and injected GPU allocation failures remain future validation work.
