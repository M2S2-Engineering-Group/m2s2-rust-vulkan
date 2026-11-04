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
    └── command.rs      # Command recording and frame synchronization

examples/
├── window.rs           # winit window only, no Vulkan
├── triangle.rs         # Indexed geometry with a fixed camera
└── cube.rs             # Two cubes with shared camera uniforms, model push constants, and depth testing
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
for the proposed diagnostic text overlay.
