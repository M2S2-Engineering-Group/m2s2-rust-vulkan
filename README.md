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
├── window.rs           # Window management with winit
├── math.rs             # Math utilities with nalgebra
└── renderer/
    ├── mod.rs          # Renderer module
    ├── instance.rs     # Vulkan instance creation
    ├── device.rs       # Physical/logical device selection
    ├── swapchain.rs    # Swapchain management
    ├── pipeline.rs     # Graphics pipeline (TODO)
    ├── buffer.rs       # Buffer management (TODO)
    └── command.rs      # Command buffer management (TODO)

examples/
├── window.rs           # Basic window creation
└── triangle.rs         # Basic Vulkan initialization
```

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

1. Implement graphics pipeline creation
2. Add vertex/index buffer management
3. Create basic triangle rendering
4. Add shader compilation
5. Implement basic scene management
