# Spec-Driven Development

How roadmap items in [ROADMAP.md](ROADMAP.md) turn into code. The point of writing a spec first — for a
solo/learning project as much as a team one — is that it forces you to state what you understand *before*
implementation reveals whether you actually understood it. A spec you can't finish writing means the concept
isn't clear yet; that's a cheaper place to discover that than mid-`unsafe` block.

This is deliberately lightweight: no new tooling, no numbering scheme, no separate issue tracker. A spec is a
markdown file living next to the code it describes.

## When to write one

Write a spec before starting work on:
- Any new file under `src/renderer/` (i.e., replacing one of the current stubs — `pipeline.rs`, `buffer.rs`,
  `command.rs` — or adding a new module for a later phase)
- Any change to an existing module's public API (new `pub fn`/`pub struct` on `VulkanInstance`,
  `VulkanDevice`, `VulkanSwapchain`, `VulkanRenderer`)
- Any architectural decision the roadmap flagged as "decide before implementing" (e.g. Phase 1's dynamic
  rendering vs. render-pass choice)

Skip it for: bug fixes, test additions, doc updates, or anything that doesn't add or change a public shape.
Use your judgment — the test is "would writing this down change what I build," not "is this technically new
code."

## Where it lives

`specs/NNN-short-name.md`, numbered sequentially (`001-graphics-pipeline.md`, `002-vertex-buffers.md`, ...).
Kept in the repo permanently, not deleted after merge — for a project with an explicit learning goal, the
spec is a record of what you understood *going in*, which is worth more once you can compare it against what
you understood coming out. If reality diverged from the spec during implementation, add a short "Outcome"
section at the bottom rather than editing the original plan away.

## Template

```markdown
# NNN: <Feature name>

**Roadmap item**: <link to the ROADMAP.md checkbox this implements>

## Goal
One or two sentences: what this adds and why it's needed now.

## Non-goals
What's explicitly out of scope for this spec (defer to a later spec / roadmap phase).

## Public API
Rust signatures — struct fields, `pub fn` signatures, trait impls. This is the contract; if you can't write
this section, you don't understand the feature's shape yet.

## Vulkan objects
Which `Vk*` objects this creates/owns/destroys, and their lifetime relative to existing objects (e.g. "owned
by `VulkanSwapchain`, destroyed in its `Drop`, must outlive X"). Name the destroy order explicitly if more
than one object is involved — this codebase has already hit one destroy-order bug
(`VUID-vkDestroySurfaceKHR-surface-01266`) from getting this wrong.

## Interactions
What existing modules this touches or depends on (e.g. "`pipeline.rs` needs `VulkanDevice::device` and the
swapchain's image format from `VulkanSwapchain`").

## Test plan
- Unit tests for pure logic (mirrors the existing pattern in `device.rs`/`swapchain.rs`/`error.rs` — see
  `CLAUDE.md`'s Quality gates section: CI has no GPU, so tests can't call real Vulkan; test the
  non-Vulkan-calling logic)
- Manual verification: what to run (`cargo run --example ...`) and what visually/behaviorally confirms it
  worked

## Open questions
Anything you're not sure about going in. Fine to leave unanswered — resolve during implementation and note
the resolution in Outcome.

## Outcome
(Filled in after implementation.) What actually happened — what matched the spec, what didn't, what you
learned.
```

## Workflow

1. Copy the template to `specs/NNN-short-name.md`, fill in everything through "Open questions."
2. Read the matching section of [LEARNING_PATH.md](LEARNING_PATH.md) if you haven't already — the spec
   should reflect that you've done the reading, not substitute for it.
3. Implement against the spec. It's fine for implementation to reveal the spec was wrong about something
   small (a signature tweak, a missed edge case) — fix the spec to match reality rather than let them drift,
   and note the change in Outcome if it's worth remembering why.
4. Before committing: run the quality gates (`cargo fmt`, `cargo clippy`, `cargo build --all-targets`,
   `cargo test --all-targets` — see [CLAUDE.md](CLAUDE.md)'s Quality gates section for what CI enforces).
5. Fill in Outcome.
6. Check off the corresponding item in `ROADMAP.md` in the same commit/PR.
7. Commit using the existing [Conventional Commits](https://www.conventionalcommits.org) format enforced by
   `.githooks/commit-msg` — e.g. `feat(pipeline): add graphics pipeline creation`.

## What this is not

- Not a replacement for the quality gates or CI — a spec doesn't relax `cargo clippy -D warnings`.
- Not a full spec-kit-style multi-stage pipeline (spec → plan → tasks as separate artifacts). One file is
  enough at this project's scale; revisit if that stops being true.
- Not required for exploratory spikes — if you're prototyping to figure out *whether* an approach works
  before committing to it, prototype first and write the spec once you know what you're actually proposing.
