# M2S2 Rust Vulkan

A Vulkan-based rendering engine written in Rust.

## Prerequisites

1. **Vulkan SDK**: Download and install from [LunarG](https://vulkan.lunarg.com/)
2. **Rust**: Install via [rustup](https://rustup.rs/)

## Project Structure

```
src/
├── lib.rs              # Main library interface
├── error.rs            # Error handling
├── math.rs             # Math utilities built on m2s2-math
└── renderer/
    ├── mod.rs          # VulkanRenderer: composes instance -> device -> swapchain
    ├── instance.rs     # Vulkan instance creation
    ├── device.rs       # Physical/logical device selection
    ├── swapchain.rs    # Swapchain management
    ├── pipeline.rs     # Dynamic-rendering graphics pipeline
    ├── buffer.rs       # Staged vertex/index buffers
    ├── mesh.rs         # Renderer-owned meshes and MeshId validation
    ├── scene.rs        # Per-object mesh, transform, texture, and material
    ├── command.rs      # Command recording and frame synchronization
    ├── text.rs         # Reusable bitmap text layout and GPU drawing
    ├── diagnostics.rs  # CPU frame timing and diagnostic formatting
    ├── image.rs        # Sampled R8/RGBA8 image upload
    └── upload.rs       # Shared one-time upload submission

examples/
├── window.rs           # winit window only, no Vulkan
├── triangle.rs         # Indexed geometry with a fixed camera
└── cube.rs             # Two cubes with depth testing and an F3 diagnostic text overlay
```

The library owns Vulkan mechanics only — window/event-loop creation stays with the caller (see
[CLAUDE.md](CLAUDE.md)'s Architecture section). `winit` is a dev-dependency used by the examples above, not a
library dependency.

## Running Examples

```bash
# Basic window (no Vulkan)
cargo run --example window

# Indexed geometry (requires Vulkan runtime)
cargo run --example triangle

# Two independently animated cubes
cargo run --example cube
```

## Development Notes

- Debug builds include Vulkan validation layers
- Release builds disable validation for performance
- Uses `ash` for low-level Vulkan bindings
- Cross-platform window management with `winit`

## Next Steps

See [ROADMAP.md](ROADMAP.md) for the full feature roadmap, [LEARNING_PATH.md](LEARNING_PATH.md) for the
education plan tied to it, and [SPEC_PROCESS.md](SPEC_PROCESS.md) for how each roadmap item gets specced
before implementation.

See [REVIEW_NOTES.md](REVIEW_NOTES.md) for correctness fixes and verification limits, and
[spec 007](specs/007-multiple-objects-and-push-constants.md) for multiple-object rendering, and [spec 008](specs/008-diagnostic-text-overlay.md)
for reusable text drawing and the diagnostic overlay. The cube example shows diagnostics by default;
press **F3** to hide/show them.

## Mesh resources

Construct with `VulkanRenderer::with_window(Arc::clone(&window), extent)`; the renderer starts with no
meshes. Upload each vertex/index pair with `create_mesh`, then put its `MeshId` in each `Object`.
Mesh and texture handles belong to the renderer that created them. Invalid or foreign handles reject
the whole frame before GPU work; `texture: None` explicitly selects the white default.

```rust,ignore
let mesh = renderer.create_mesh(&vertices, &indices)?;
let objects = [Object {
    mesh,
    model: Mat4::IDENTITY,
    texture: None,
    specular: SpecularMaterial::default(),
}];
renderer.render_objects(&objects, camera)?;
// Single-object convenience path:
renderer.render_frame(mesh, Mat4::IDENTITY, camera)?;
```

Upload different meshes through the same renderer, or reuse a handle across many objects. Meshes stay
alive until renderer teardown. Empty object slices can still clear/present and draw text without any
mesh upload. The raw-handle `unsafe new` constructor also takes only handles and extent; geometry is
uploaded afterward. See [spec 015](specs/015-multiple-mesh-resources.md) for validation and ownership.

## Drawing text

Text positions use physical pixels from the top-left; scale multiplies an 8×8 bitmap font.
Supply text each frame; an empty slice hides it. Existing single/multiple-object calls still work.

```rust,ignore
use m2s2_rust_vulkan::{Camera, TextDraw};
use m2s2_rust_vulkan::math::{Vec2, Vec4};

// Keep Diagnostics between frames; the caller measures CPU frame interval and owns the camera.
diagnostics.update(frame_interval);
let message = diagnostics.format(&renderer.stats(), camera_position);
let text = [TextDraw {
    text: &message,
    position: Vec2::new(12.0, 12.0),
    scale: 2.0,
    color: Vec4::new(1.0, 1.0, 1.0, 1.0),
}];
let camera = Camera {
    view,
    proj,
    position: camera_position,
};
renderer.render_objects_with_text(&objects, camera, &text)?;
```

`TextDraw` accepts ordinary labels too; it has no dependency on diagnostics. The renderer supports
newlines and ASCII, substitutes `?` for unsupported characters, clips at the viewport, and reports an
error above 2048 visible glyphs. CPU timing is application frame cadence, not GPU execution time.
