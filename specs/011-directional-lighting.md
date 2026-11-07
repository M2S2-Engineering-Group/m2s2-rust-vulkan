# 011: Directional lighting

**Status**: Implemented (CPU gates pass; GPU acceptance pending user verification).
**Roadmap item**: [Phase 2](../ROADMAP.md#phase-2-standard-rendering-features), "Basic lighting" —
this spec implements the **directional + ambient + diffuse** slice of that item. Specular (Phong/
Blinn-Phong) and point/spot lights are deferred to follow-up specs; the roadmap box stays unchecked
until those land (see Non-goals and Outcome).
**Prerequisite**: specs 006–010 implemented. Builds on the per-frame mapped-uniform path
(`src/renderer/uniform.rs`), the per-object `Object`/texture draw path (spec 010), and the model push
constant (spec 007).

## Goal

Light scene objects with a single stationary **directional ("sun") light** so that faces oriented
toward the light are bright and faces oriented away fall off into a dim ambient floor — standard
per-pixel **Lambert diffuse + ambient** shading, multiplied with the existing texture × vertex-color.
This is the first real shading model in the engine and the first use of surface normals.

## Non-goals

- **No specular** (no Phong/Blinn-Phong highlight term). Diffuse + ambient only. Specular needs the
  camera/eye position in the fragment stage and a material shininess; deferred to a follow-up spec.
- **No point or spot lights** in this spec. The light data and shader are *structured* so a light
  *type* with a position/range/cone extends them later, but only the directional path is implemented
  now. ("Flashlight, lamp, etc." from the design discussion are explicitly future work.)
- **No multiple lights.** One directional light. Multi-light (an array + count, or a light list) is a
  later extension of the same uniform.
- **No shadows** (shadow maps are Phase 4), no normal mapping, no HDR/tone-mapping, no per-material
  shininess/roughness.
- **No camera controller in the library.** The free-fly camera, reset key, and on-screen camera
  position are demonstration-only, live in `examples/cube.rs`, and add no library API — camera
  controllers remain engine-layer/consumer concerns (consistent with the library staying scoped to
  Vulkan mechanics). See "Demonstration (example only)".

## Public API

A directional light is a small, ergonomic value type (no Vulkan padding exposed). The renderer stores
the current light and uploads it each frame, so configuring it is a setter, not a per-call argument —
the existing `render_*` signatures are unchanged. A sensible default sun is set at construction, so
objects are lit without the caller doing anything. Re-export the light type from the crate root.

```rust
/// A single directional ("sun") light in world space. Ergonomic value type; the renderer converts it
/// to the std140 GPU layout internally.
#[derive(Copy, Clone, Debug)]
pub struct DirectionalLight {
    /// The direction the light travels (sun rays), world space. Need not be normalized; the shader
    /// normalizes. The surface-to-light vector used for shading is `-direction`.
    pub direction: Vec3,
    /// Diffuse light color/intensity. White (`1,1,1`) is a neutral full-strength sun.
    pub color: Vec3,
    /// Ambient color added to every fragment regardless of orientation, so the unlit side is dim
    /// rather than pure black. Typically a small fraction of `color`.
    pub ambient: Vec3,
}

impl Default for DirectionalLight {
    /// A neutral sun from the upper-front-right with a mild ambient floor. Exact values documented in
    /// the implementation; chosen so the stock cube is clearly lit on exactly the faces toward the sun.
    fn default() -> Self;
}

impl VulkanRenderer {
    /// Replace the directional light used for subsequent frames. Takes effect on the next `render_*`
    /// call (uploaded into that frame's light uniform). Cheap; may be called every frame if a future
    /// consumer wants a moving light.
    pub fn set_directional_light(&mut self, light: DirectionalLight);
}
```

The `render_frame` / `render_objects` / `render_objects_with_text` signatures are **unchanged** from
spec 010 — lighting is applied to whatever they draw, using the currently-set light.

## Vulkan objects

### Vertex layout (breaking change)
`Vertex` gains `normal: Vec3` **after** `uv` (preserving existing field order): `position` (offset 0),
`color` (offset 12), `uv` (offset 24), `normal` (offset 32), size **44**, still tightly packed
(`Vec3`/`Vec2` are `[f32; N]`-layout `Pod`). `attribute_descriptions` returns **four** entries, adding
`location = 3`, `R32G32B32_SFLOAT` at `offset_of!(Vertex, normal)`. This is an intentional pre-1.0
break, same shape as the `uv` addition in spec 010: every consumer now supplies a per-vertex normal.

### Light uniform descriptor set (new set 2)
Camera stays at **set 0** (binding 0, vertex stage); per-object texture stays at **set 1** (spec 010).
Add a **new set 2**, binding 0: a `UNIFORM_BUFFER` visible to the **fragment** stage holding the light.
Rationale for a dedicated set rather than extending the camera set: it is additive (the working camera
uniform path is not modified — open/closed), it mirrors the self-contained module pattern established
by `texture.rs`, and it binds once per frame like the camera (vs. set 1 which binds per object).

The set is **per-frame-in-flight** (one uniform buffer + one descriptor set per frame, host-visible and
mapped), identical in lifetime/update discipline to the camera uniform: the CPU writes frame N's light
buffer only after waiting on frame N's fence, so a future per-frame-changing light has no GPU-read race.
For this spec the light is stationary, but the per-frame structure keeps the door open and reuses the
existing pattern.

The std140 GPU block is three `vec3`s, each 16-byte aligned:

```glsl
layout(set = 2, binding = 0) uniform LightUniforms {
    vec3 direction;   // offset 0
    vec3 color;       // offset 16
    vec3 ambient;     // offset 32
} light;              // size 48
```

The Rust mirror is a `#[repr(C)]` `Pod` struct with explicit trailing-`f32` padding after each `Vec3`
to match those offsets (size 48, offsets 0/16/32), verified by a layout unit test exactly like
`camera_uniforms_match_shader_layout`. The public `DirectionalLight` is converted into this padded
struct at upload time; callers never see the padding.

### Shared per-frame mapped-uniform helper
The light uniform is the *second* instance of the "per-frame-in-flight, host-visible, mapped, single
binding" pattern (the camera uniform is the first). Per the shared-before-one-off rule, extract the
guts of `uniform.rs` into a small reusable helper that owns the per-frame layout + pool + sets +
mapped buffers for one binding, parameterized by the POD payload type, the binding index, and the
shader stage. The camera (`CameraUniforms`, binding 0, vertex, set 0) and the light (`LightUniforms`,
binding 0, fragment, set 2) are both thin users of it. This is additive refactoring driven by the new
requirement, not gratuitous churn — see Interactions for the exact split and the "builder's call" note
on how far to generalize.

### Normal transformation
The vertex shader transforms each normal to world space with the **normal matrix**
`transpose(inverse(mat3(model)))` derived from the existing model push constant — correct under
non-uniform scale, and self-contained (no new push-constant or uniform data). For the stock cube
(rotation + translation only) `mat3(model)` would suffice, but the inverse-transpose is the correct
general form and the per-vertex cost is negligible here. Precomputing it on the CPU and passing it as
push/uniform data is a documented future optimization, not part of this spec.

### Pipeline layout
The **scene** pipeline layout's set-layout array grows to `[camera(set 0), texture(set 1), light(set
2)]`. The **text** pipeline is unaffected (its own layout, its own set 0 font atlas, its own shaders).

## Recording

Per frame, after binding the camera descriptor set at `firstSet = 0`, bind the current frame's light
descriptor set once at `firstSet = 2` (alongside the existing per-frame camera bind, before the object
loop). The per-object set-1 texture bind inside `emit_object_draws` is unchanged. The light buffer for
the frame is written during the frame's uniform-update step, next to the camera write
(`uniforms.update(...)`), both after the in-flight fence wait.

## Shaders

`shaders/triangle.vert`:
- add `layout(location = 3) in vec3 inNormal;`
- add `layout(location = 2) out vec3 fragNormal;` (keep `fragColor = 0`, `fragUV = 1`)
- compute `mat3 n = transpose(inverse(mat3(object.model))); fragNormal = normalize(n * inNormal);`

`shaders/triangle.frag`:
- add `layout(location = 2) in vec3 fragNormal;`
- add the set-2 `LightUniforms` block above
- shade:
  ```glsl
  vec3 N = normalize(fragNormal);
  vec3 L = normalize(-light.direction);            // surface -> light
  float diff = max(dot(N, L), 0.0);                // Lambert
  vec3 shade = light.ambient + light.color * diff; // ambient floor + diffuse
  vec4 albedo = texture(tex, fragUV) * vec4(fragColor, 1.0);
  outColor = vec4(shade, 1.0) * albedo;
  ```

Recompile with the existing `glslc` workflow, commit the `.spv`, and validate with `spirv-val`. The
scene SPIR-V magic-word parse tests must still pass against the recompiled binaries.

## Ownership and teardown

The light descriptor set/layout/pool/buffers are owned by the renderer (through the shared per-frame
helper), with the same lifetime and explicit teardown as the camera uniform: freed in the renderer's
`Drop` within the existing ordered teardown — mapped buffers freed via the allocator before the
allocator is dropped, layout/pool destroyed via the device, all before the device. Construction follows
the existing partial-init rollback pattern (hold in `Option`s / `Cleanup` guards so an error mid-`new`
unwinds the light resources). No caller-visible resource, nothing to leak or dangle.

## Interactions

- `buffer.rs`: add `normal` to `Vertex` and a fourth attribute description; update the doc comment and
  the packing/attribute tests (size 44, offset 32, `location = 3`).
- `uniform.rs`: factor its per-frame mapped-uniform mechanics into the shared helper; `CameraUniforms`
  becomes a user of it. Public behavior of the camera path is unchanged.
- new `light.rs` (or equivalent): `DirectionalLight` (public), the padded `LightUniforms` (internal),
  the set-2 layout constant/stage, and the light's per-frame uniform (via the shared helper).
- `pipeline.rs`: scene pipeline layout gains `set_layouts[2]` (the light layout). Text pipeline
  unchanged. (Both the initial build and the resize-path rebuild pass the three-layout array.)
- `command.rs`: `draw_frame`/`record` take the light's per-frame descriptor set + the current
  `LightUniforms` to upload; bind set 2 once per frame; write the light buffer next to the camera.
- `mod.rs`: own the light resources and the current `DirectionalLight` (default at construction);
  `set_directional_light`; thread the light into the frame; teardown additions; pass the three-layout
  array to the pipeline in both `new` and the resize path.
- `lib.rs`: re-export `DirectionalLight`.
- `shaders/triangle.{vert,frag}` + committed `.spv`.
- `examples/cube.rs`: rebuild the cube with per-face normals (see below); add the free-fly camera,
  reset key, and optionally a `set_directional_light` call to pose the sun.
- `examples/triangle.rs`: add a normal (`(0, 0, 1)`, facing the camera) to each of its 3 vertices so it
  compiles and is lit by the default sun. Still uses `render_frame`.

## Demonstration (example only)

`examples/cube.rs`, no library API:

- **Per-face normals**: rebuild the cube from 8 shared vertices to **24 vertices** (4 per face), each
  face's four vertices carrying that face's outward normal, with 36 indices (6 per face). A corner
  shared by three faces cannot carry three normals, so flat per-face shading requires the split — this
  is consumer geometry, not a library concern.
- **Free-fly camera**: track `position`, `yaw`, `pitch`; **WASD** moves (forward/back/strafe relative
  to facing), **arrow keys** look (left/right yaw, up/down pitch, pitch clamped to ±89°). Movement and
  look are scaled by the frame delta already available from the diagnostics frame-interval, so speed is
  frame-rate independent. View is `look_at(position, position + forward, up)`. (Keyboard-only; true
  mouse-look with cursor grab is intentionally out of scope.)
- **Reset**: **R** restores the camera to its initial pose.
- **On-screen camera position**: already rendered by the diagnostics overlay (`Camera: (x, y, z)`); it
  starts updating once the camera moves. No overlay/library change.
- The stationary sun makes the moving camera reveal the lit/unlit faces from every angle — the intended
  "lighting like real life" demonstration.

## Test plan

CPU-only (CI has no driver):
- `Vertex`: size 44; offsets position 0, color 12, uv 24, normal 32; four attribute descriptions, the
  fourth `location = 3`, `R32G32B32_SFLOAT`, offset 32.
- `LightUniforms` std140 layout: size 48, offsets direction 0, color 16, ambient 32 (mirrors
  `camera_uniforms_match_shader_layout`).
- `DirectionalLight` → `LightUniforms` conversion: fields land at the right offsets; `Default` produces
  the documented sun.
- The shared per-frame-uniform helper: whatever pure logic it exposes (e.g. frame indexing) keeps its
  existing camera-side test coverage; no regression in the camera layout test.
- Scene vertex/fragment SPIR-V parse (magic word) after recompilation.

Manual GPU checks (with `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`):
- The cubes show clearly brighter faces toward the sun and dim (ambient-floor) faces away from it;
  rotating the view with the free-fly camera moves which faces are lit, consistent with a fixed world
  light.
- Textured faces still show their texture, now modulated by the lighting; the default-white/untextured
  path still works (albedo = vertex color × white, then shaded).
- Depth occlusion unchanged; WASD/arrow movement and **R** reset behave; the overlay camera position
  tracks the camera.
- Repeated resize, minimize/restore, and shutdown produce no validation errors; teardown leaks nothing
  (gpu-allocator clean on drop).

Run all four repository quality gates. Do **not** check the roadmap "Basic lighting" box in this spec —
it also covers specular and point/directional multiplicity; note in Outcome what fraction is done and
leave the box for the follow-up that finishes it. (If preferred, convert the single roadmap line into a
checklist and check only the directional/diffuse/ambient/normals sub-items — decide in Outcome.)

## Open questions

- **How far to generalize the shared per-frame-uniform helper**: minimum is "single-binding, one POD
  payload, parameterized stage" serving camera and light. Whether it should also support multi-binding
  sets (anticipating a future that groups globals) is a builder's call — prefer the minimal version
  that removes the duplication without speculative surface, and note the decision in Outcome.
- **Default sun values** (direction, color, ambient magnitude): pick concrete values during
  implementation that make the stock cube visibly and pleasantly lit; record them in Outcome.

## Outcome

Implemented across four tasks (`5e89b92`..`feat(examples)` cube demo): `Vertex` gained `normal`
(location 3, offset 32, size 44); `uniform.rs`'s mapped-UBO machinery was extracted into a shared
`per_frame_uniform::PerFrameUniform` (parameterized by payload size, binding, stage), which both the
camera (`Uniforms`) and the new `light.rs` (`Light`) wrap; `light.rs` added the public
`DirectionalLight` (with a `Default` sun), the internal std140 `LightUniforms`, and the set-2 light
uniform (fragment stage); the scene pipeline layout grew to `[camera(0), texture(1), light(2)]`;
`command.rs` binds set 2 once per frame and uploads the light next to the camera; `mod.rs` owns the
light, defaults it, exposes `set_directional_light`, and tears it down in `Drop`; the shaders transform
normals by the inverse-transpose normal matrix and shade `(ambient + color·max(dot(N,L),0)) × albedo`.
The cube example rebuilt its mesh to 24 per-face vertices and added a free-fly camera (WASD + arrow
keys, **R** to reset) that poses a stationary sun.

Design decisions (builder's calls): the light lives in its own descriptor set (2), not grafted onto the
camera set (additive/open-closed); configured via a stored-light setter rather than a per-call argument
(render signatures unchanged); the shared per-frame helper is the minimal single-binding form. Default
sun: `direction (-0.5, -1.0, -0.3)`, `color (1,1,1)`, `ambient (0.15, 0.15, 0.15)`.

Verified (CPU): 51 unit tests pass (incl. `Vertex` size/offsets with `normal`, `LightUniforms` std140
layout 48/0/16/32, `DirectionalLight`→`LightUniforms` conversion and `Default`, and SPIR-V magic-word
parse of the recompiled shaders); all four quality gates clean; `glslc` + `spirv-val` pass on both
shaders. **GPU acceptance (Tasks 3 Step 10, 4 Step 5) could not run in the sandbox (no Vulkan driver)
and is pending a user run of `cargo run --example cube`** — the per-frame set-2 bind and the
normal-matrix shading are only observable there.

Deferred per Non-goals: specular (Phong/Blinn-Phong), point/spot lights, multiple lights, shadows. The
roadmap "Basic lighting" box therefore stays unchecked pending that follow-up work.
