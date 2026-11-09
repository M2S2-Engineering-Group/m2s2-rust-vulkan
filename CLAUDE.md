# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Vulkan rendering engine library in Rust, built on `ash` (raw Vulkan bindings). It is being developed as a
reusable rendering backend for two separate consumers: a Vulkan tutorial project, and a port of an old C++
college project. Because of that, the library is intentionally scoped to Vulkan mechanics only — it does not
own window creation or the event loop; callers bring their own (see Architecture below).

It depends on `m2s2-math` (linear algebra: vectors, matrices, quaternions), published on crates.io.

## Commands

```bash
cargo build              # build the library
cargo build --examples   # build the examples (catches drift between lib API and example usage)
cargo run --example window    # winit window only, no Vulkan — sanity-checks the windowing setup
cargo run --example triangle  # minimal end-to-end: one static triangle under a fixed camera
cargo run --example cube      # 3D: per-frame MVP uniforms, a independent model push constants, depth testing
cargo test                # unit tests (currently minimal — see Status)
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Quality gates

`cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build
--all-targets`, and `cargo test --all-targets` must all pass clean (zero warnings) — this is enforced both
locally and in CI:

- **Pre-commit hook**: `.githooks/pre-commit` runs all four. Enable it once per clone with
  `git config core.hooksPath .githooks`. It is not enabled by default — a fresh clone has no active hook until
  you run that command.
- **CI**: `.github/workflows/ci.yml` runs the same four checks on every push/PR to `main`, on `ubuntu-latest`.
  There's no GPU or Vulkan SDK on the runner — this is fine because `ash`'s default `loaded` feature dynamically
  loads the Vulkan library at runtime (via `libloading`), so building and running the (non-Vulkan-calling) unit
  tests doesn't require Vulkan to actually be present. Don't add a test that calls `Entry::load()` or otherwise
  touches a real Vulkan driver without first giving CI a software implementation (e.g. lavapipe) or gating it
  behind a feature/env check.

There's no rust-toolchain pin — CI and local dev both use whatever stable toolchain is installed.

### Headless GPU tests and golden images

`src/renderer/headless.rs` and `src/renderer/golden.rs` (both `#[cfg(test)]`-only modules) exercise real
Vulkan device creation, offscreen rendering, and pixel-level output against a reference PNG, none of which
CI's driver-less runner can do (see Quality gates above). Those tests are `#[ignore]`d by default so
`cargo test --all-targets` stays green on every machine; running them requires a real Vulkan driver.

- Run the full ignored set on a machine with a GPU/driver: `cargo test -- --ignored`.
- `self_test_scene_matches_golden_with_zero_validation_errors` (in `renderer::headless`) compares its
  rendered frame against `testdata/golden/headless_self_test.png` via `assert_matches_golden` (per-channel
  tolerance, see `golden.rs`). To create or regenerate that golden image after an intentional rendering
  change, run the same test with `M2S2_UPDATE_GOLDEN=1` set — this makes the comparison write the actual
  output as the new golden instead of diffing against it — then re-run without the env var to confirm the
  test now passes against what you just wrote, and commit the resulting PNG under `testdata/golden/`:
  ```bash
  M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless -- --ignored self_test_scene_matches_golden_with_zero_validation_errors
  cargo test --lib renderer::headless -- --ignored self_test_scene_matches_golden_with_zero_validation_errors
  ```

### Commit message format (required for releases)

Commit subjects must follow [Conventional Commits](https://www.conventionalcommits.org):
`<feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert>(<scope>)?!?: <description>`. This is enforced
by `.githooks/commit-msg` (same `core.hooksPath` setup as above) and by the `commit-lint` job in
`ci.yml` — both call the shared `scripts/check-commit-msg.sh`. This isn't just style: release-plz (below)
determines the next version bump by parsing these prefixes, so an unconventional commit message silently
produces no release rather than a failed build.

### Releases (release-plz)

`.github/workflows/release-plz.yml` runs [release-plz](https://release-plz.dev/) on every push to `main`:
it opens/updates a "Release-plz" PR that bumps `Cargo.toml`'s version and writes `CHANGELOG.md` from the
Conventional Commits merged since the last release. Merging that PR triggers the second job, which cuts a
git tag and a GitHub Release. `release-plz.toml` sets `publish = false` — crates.io publishing is
deliberately not wired up yet (see Status); no `CARGO_REGISTRY_TOKEN` secret exists. When the engine has
real functionality worth shipping, flip `publish = true`, fill in the crates.io-required Cargo.toml metadata
(`license`, `description`, `repository`), and add that secret.

**Auth uses a GitHub App, not the default `GITHUB_TOKEN`.** The org has "Allow GitHub Actions to create and
approve pull requests" disabled, so the default token gets a 403 (`GitHub Actions is not permitted to create
or approve pull requests`) when release-plz tries to open the release PR — this restriction applies
regardless of the workflow's `permissions:` block. A GitHub App installation token isn't subject to it, and
as a side benefit, PRs/pushes made with an App token (unlike the default `GITHUB_TOKEN`) actually trigger
downstream workflows, so `ci.yml` runs on the release-plz PR instead of silently not firing.

Both jobs in `release-plz.yml` call `actions/create-github-app-token@v1` to mint a short-lived token from:
- `secrets.APP_ID` — the App's numeric ID
- `secrets.APP_PRIVATE_KEY` — the App's PEM private key

The App must be installed on this repo with `contents: write` and `pull-requests: write` permissions.

### Dependency on `m2s2-math`

`Cargo.toml` depends on the published `m2s2-math = "0.2"` from crates.io — CI and `cargo publish` both resolve
it from there, so CI is testing exactly what a downstream consumer would get. When `m2s2-math`'s public API
changes (e.g. a method rename), `src/math.rs` in this crate needs a matching update after bumping the version
requirement — this has already caused a silent build break once (see Status).

For live cross-repo development against an unpublished sibling checkout, create an untracked
`.cargo/config.toml` (gitignored, not committed — CI must not see it) with:

```toml
paths = ["../m2s2-rust-math"]
```

This overrides `m2s2-math` to resolve from the local sibling path instead of crates.io, as long as the sibling
repo exists at `../m2s2-rust-math` relative to this one. Caveat: cargo warns (currently non-fatal, planned to
become a hard error) if the sibling's own dependency set has drifted from what's published — check `cargo
build` output if this override starts misbehaving.

### Vulkan binding choice: `ash`, not `vulkanalia`

Evaluated switching to `vulkanalia` (the binding built for the Rust port of vulkan-tutorial.com) since one of
the two intended consumers is following that tutorial. Decided to stay on `ash`: it has a much larger
ecosystem of complementary crates (`gpu-allocator` for the buffer/memory work still to come, `ash-window`,
and it's what `wgpu`'s Vulkan backend uses), and a broader maintainer base than vulkanalia's effectively
single-maintainer project. vulkanalia's extension-trait ergonomics (no manually-constructed loader structs)
are nicer, but not enough to justify redoing `instance.rs`/`device.rs`/`swapchain.rs` again right after the
0.38 migration.

## Architecture

`src/renderer/` is a thin layering of Vulkan objects, each wrapping the next:

- **`instance.rs`** — `VulkanInstance`: `Entry` + `ash::Instance`, plus a validation/debug messenger under
  `cfg(debug_assertions)` only (release builds skip it). Owns the one `create_surface` implementation
  (`ash_window::create_surface`) — `device.rs` and `swapchain.rs` both call `instance.create_surface(...)`
  rather than each having their own copy.
- **`device.rs`** — `VulkanDevice`: enumerates all physical devices and scores them (`score_device_type`:
  discrete > integrated > virtual > CPU), skipping any that lack a graphics+present queue, the
  `VK_KHR_swapchain` extension, or a non-empty format/present-mode list for the surface. Creates a transient
  surface purely to run that check, then destroys it — the swapchain's real surface is separate and
  longer-lived (see below).
- **`swapchain.rs`** — `VulkanSwapchain`: surface, swapchain, and image views. Owns its surface for its full
  lifetime (`surface`/`surface_loader` fields) and destroys the swapchain before the surface in `Drop` —
  destroying them in the other order is a Vulkan spec violation (`VUID-vkDestroySurfaceKHR-surface-01266`)
  that the original code hit (the surface was destroyed right after creating the swapchain from it).
- **`pipeline.rs`, `buffer.rs`, `command.rs`** — dynamic-rendering pipeline, staged indexed geometry,
  command recording, frame fences, and per-image present semaphores.
- **`mod.rs`** — `VulkanRenderer` composes Vulkan resources, draws caller-provided geometry with camera
  uniforms and per-draw model push constants and depth testing, and handles resize/minimize. `Drop` waits for idle before ordered teardown.
- **`depth.rs`, `uniform.rs`, `allocator.rs`** — shared synchronized depth attachment, per-frame mapped
  uniforms/descriptors, and GPU suballocation. Private `cleanup.rs` provides constructor rollback helpers.

**Window handles, not window ownership**: the renderer takes handles (`RawDisplayHandle` +
`RawWindowHandle`, from `raw-window-handle`, a direct dependency) plus an explicit `(u32, u32)` surface
extent — not a `Window` type. `winit` is a dev-dependency only, used by the examples to obtain those
handles. This split is deliberate: window/event-loop ownership stays with whichever application embeds
the renderer, so the two intended consumers (which may already have their own main loops) aren't forced
into this crate's choices.

Spec 012 added **two constructors** over this split (see [spec 012](specs/012-resource-ownership-and-memory-safety.md)):
`VulkanRenderer::with_window(Arc<W: HasWindowHandle + HasDisplayHandle>, …)` is the **safe default** —
the renderer retains the window owner so the surface's window-lifetime contract holds by construction;
and `unsafe fn new(RawDisplayHandle, RawWindowHandle, …)` is the raw-handle escape hatch for consumers
that own their own loop and don't want the renderer holding a window reference. The safe path softens
"no window ownership" only for callers that opt into it. `VulkanRenderer`'s fields and the low-level
owners (`VulkanInstance`/`VulkanDevice`/`VulkanSwapchain`/…) are private — the `renderer` module is not
`pub`; only the data types and `VulkanRenderer` are re-exported from the crate root.

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
- Switched `m2s2-math` from a local path dependency to the published crates.io version, added a pre-commit
  hook and a GitHub Actions CI workflow (fmt + clippy + build + test), and fixed the codebase to actually pass
  those gates (formatting, unused imports, inlined format args, C-string literals) — see Quality gates.
- Added Conventional Commit enforcement (hook + CI) and release-plz for automated versioning, changelog, and
  git tags/GitHub Releases on merge to `main` — crates.io publishing intentionally left off for now.
- Added real physical device suitability scoring (`device.rs`), consolidated surface creation into
  `VulkanInstance::create_surface` (was duplicated in `device.rs` and `swapchain.rs`), fixed the
  destroy-order bug that violated `VUID-vkDestroySurfaceKHR-surface-01266`, added Wayland alongside Xlib as
  a Linux surface extension, removed the stale `MATH_TODO.md`, and added unit tests for the pure logic in
  `device.rs`/`swapchain.rs`/`error.rs`/`math.rs` (13 tests total, up from 1).

### Current rendering status

Phase 1 and Phase 2 depth/uniform/camera support are implemented, including `examples/cube.rs`.
`ROADMAP.md` tracks remaining features. See [REVIEW_NOTES.md](REVIEW_NOTES.md) for the correctness audit,
its fixes and verification limits. [Spec 007](specs/007-multiple-objects-and-push-constants.md) implements
multiple independently transformed copies of one mesh using push constants. `render_objects` accepts a
model slice; `render_frame` remains a single-object compatibility wrapper. [Spec 008](specs/008-diagnostic-text-overlay.md)
implements reusable per-frame text drawing with a separate CPU diagnostic formatter. The cube example
shows camera/FPS/device/swapchain/scene metrics, with F3 visibility control. Shared `image.rs` and
`upload.rs` supply sampled-R8 uploads and one-time submission; scene/text pipelines share construction.
[Spec 009](specs/009-vulkan-1-3-core-dynamic-rendering.md) raised the required API to Vulkan 1.3 so
dynamic rendering is core: the `VK_KHR_dynamic_rendering` extension and its pre-1.2 dependency chain,
the `get_physical_device_properties2` instance extension, and both the properties2 and
dynamic-rendering loaders are gone; device selection now gates on a `>= 1.3` version check and
`cmd_begin/end_rendering` are core `ash::Device` calls.
