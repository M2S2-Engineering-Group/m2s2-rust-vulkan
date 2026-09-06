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
    ├── pipeline.rs     # Graphics pipeline (TODO)
    ├── buffer.rs       # Buffer management (TODO)
    └── command.rs      # Command buffer management (TODO)

examples/
├── window.rs           # winit window only, no Vulkan
└── triangle.rs         # Vulkan instance/device/swapchain init + render loop stub
```

The library owns Vulkan mechanics only — window/event-loop creation stays with the caller (see
[CLAUDE.md](CLAUDE.md)'s Architecture section). `winit` is a dev-dependency used by the examples above, not a
library dependency.

## Running Examples

```bash
# Basic window (no Vulkan)
cargo run --example window

# Vulkan initialization (requires Vulkan SDK)
cargo run --example triangle
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
