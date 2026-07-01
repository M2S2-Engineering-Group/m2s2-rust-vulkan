# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Vulkan rendering engine library in Rust, built on `ash` (raw Vulkan bindings). It is being developed as a
reusable rendering backend for two separate consumers: a Vulkan tutorial project, and a port of an old C++
college project. Because of that, the library is intentionally scoped to Vulkan mechanics only — it does not
own window creation or the event loop; callers bring their own (see Architecture below).

It depends on a sibling crate, `m2s2-math` (linear algebra: vectors, matrices, quaternions), via a local path
dependency — not from crates.io.

## Commands

```bash
cargo build              # build the library
cargo build --examples   # build the examples (catches drift between lib API and example usage)
cargo run --example window    # winit window only, no Vulkan — sanity-checks the windowing setup
cargo run --example triangle  # Vulkan instance/device/swapchain init + render loop stub
cargo test                # unit tests (currently minimal — see Status)
```

There is no CI, clippy config, rustfmt config, or rust-toolchain pin in this repo. `cargo build`,
`cargo build --examples`, and `cargo test` all passing is the bar for a change being done.

### Sibling dependency

`Cargo.toml` points `m2s2-math` at `../m2s2-rust-math` (relative to this repo, not nested inside it). This repo
cannot build standalone without that sibling checked out at that path. When `m2s2-math`'s public API changes
(e.g. a method rename), `src/math.rs` in this crate needs a matching update — this has already caused a build
break once (see Status).

## Architecture

`src/renderer/` is a thin layering of Vulkan objects, each wrapping the next:

- **`instance.rs`** — `VulkanInstance`: `Entry` + `ash::Instance`, plus a validation/debug messenger under
  `cfg(debug_assertions)` only (release builds skip it).
- **`device.rs`** — `VulkanDevice`: picks a physical device (currently just `physical_devices[0]`, no
  suitability scoring), resolves graphics/present queue families, creates the logical device + queues.
- **`swapchain.rs`** — `VulkanSwapchain`: surface, swapchain, and image views.
- **`pipeline.rs`, `buffer.rs`, `command.rs`** — stub placeholders (`TODO`), not yet implemented.
- **`mod.rs`** — `VulkanRenderer` composes instance → device → swapchain. `render_frame()` is currently a
  no-op stub; `Drop` calls `wait_idle()`.

**Window handles, not window ownership**: `VulkanRenderer::new`, `VulkanDevice::new`, and
`VulkanSwapchain::new` take `RawDisplayHandle` + `RawWindowHandle` (from `raw-window-handle`, a direct
dependency) plus an explicit `(u32, u32)` surface extent — not a `Window` type. `winit` is a dev-dependency
only, used by the examples to obtain those handles. This split is deliberate: window/event-loop ownership
stays with whichever application embeds the renderer, so the two intended consumers (which may already have
their own main loops) aren't forced into this crate's choices.

One known wart: `device.rs` and `swapchain.rs` each independently create and destroy their own throwaway
surface (device.rs needs one transiently to query present support; swapchain.rs creates its own real one).
This works but duplicates surface-creation logic — not yet consolidated.

`src/math.rs` is a thin convenience layer over `m2s2-math`: type aliases (`Mat4`, `Vec2/3/4`, `Point2f/3f`)
and free functions (`perspective`, `look_at`, `translate`, `rotate`) that call `m2s2-math`'s
Vulkan-convention methods specifically (`perspective_rh_zo`, `look_at_rh`) — `m2s2-math` also exposes OpenGL/
D3D conventions (`_rh_no`, `_lh_zo`, etc.) that this layer does not surface.

## Status

### Done
- Fixed the `m2s2-math` path dependency (was pointing at a nonexistent nested directory).
- Fixed `src/math.rs` calling `m2s2-math` methods that had been renamed (`perspective`/`look_at` →
  `perspective_rh_zo`/`look_at_rh`) — this was a silent build break from the two repos drifting out of sync.
- Migrated `ash` 0.37 → 0.38 and `ash-window` 0.12 → 0.13 (builder-pattern API → `default()` + chained
  setters; module paths like `ash::extensions::khr::Surface` → `ash::khr::surface`) to resolve a
  `raw-window-handle` version conflict between `ash-window` (pinned to rwh 0.5) and `winit` 0.29 (rwh 0.6).
- Removed window/event-loop ownership from the library (see Architecture) and updated both examples for
  winit 0.29's actual event-loop API (2-arg closure, `Event::AboutToWait`, `elwt.exit()` — the examples had
  been written against an older winit API and `cargo build --examples` was silently broken before any of
  this).

### Not yet done
Tracked informally in `README.md`'s "Next Steps" and `LEARNING_PATH.md`; the stub files under `src/renderer/`
are the concrete markers:
- Graphics pipeline creation (`pipeline.rs`)
- Vertex/index buffer management (`buffer.rs`)
- Command pool + command buffer management (`command.rs`)
- An actual render loop — `render_frame()` is a no-op; acquiring an image, recording/submitting a command
  buffer, and presenting are all unimplemented
- Swapchain recreation on window resize (the `triangle.rs` resize handler is a `TODO`)
- Physical device suitability scoring (currently always picks device 0)
- Shader loading/SPIR-V compilation
- `MATH_TODO.md` is stale — it lists math functions as not-yet-implemented that already exist in `m2s2-math`
  under different (convention-suffixed) names; it should be reconciled or removed rather than trusted as-is.
- Linux surface creation is hardcoded to Xlib (`ash::khr::xlib_surface`) — no Wayland path, so this won't work
  under a Wayland session.
