# 018: Bounded point lighting

**Status**: Draft for review; not implemented.
**Roadmap item**: Phase 2 basic lighting, point-light and multiple-light portion.
**Dependencies**: Spec 011 directional light and validation, 012 resource safety, and 013's frame Camera
and per-object SpecularMaterial. Mesh resources/instancing/mipmaps are independent design extensions.
If 017 is present, its visibility applies only to the directional light, not these point lights.

## Goal

Add up to eight positioned colored lights with finite range and smooth distance falloff, alongside
the existing directional light. Reuse the same world-space diffuse/specular material response.

## Non-goals

Spot lights, point-light shadows, physically based inverse-square photometry, clustered/tiled/deferred
lighting, unbounded light lists, per-light ambient terms, cookies, a general light-type registry, or
an editable lighting UI. This is a small forward-rendered light list.

## Public API

```rust
pub const MAX_POINT_LIGHTS: usize = 8;

#[derive(Copy, Clone, Debug)]
pub struct PointLight {
    pub position: Vec3, // world space, finite
    pub color: Vec3,    // finite nonnegative linear intensity, values above one allowed
    pub range: f32,     // world units, finite and > 0
}

impl VulkanRenderer {
    pub fn set_point_lights(&mut self, lights: &[PointLight]) -> Result<()>;
}
```

Re-export PointLight and MAX_POINT_LIGHTS. Default list is empty; setting an empty slice removes all
point lights. Copy the complete validated list into renderer-owned CPU state. Reject more than eight,
non-finite positions/intensities, negative intensity, or nonpositive/non-finite range before changing
any current light state. Never silently truncate. Rejection preserves the entire previous list.

The existing fallible directional-light setter stays independent. To demonstrate point lights alone,
use zero directional diffuse and ambient colors while keeping its direction valid. No zero-direction
shortcut or implicit replacement of the directional light.

## Shading contract

For fragment world position P and point position Q, distance d = length(Q-P), direction L = (Q-P)/d.
Use the shared 013 response for Lambert diffuse plus Blinn–Phong specular. Proposed falloff:

```text
if d >= range: contribution = 0
otherwise:
    t = 1 - d / range
    attenuation = t * t
    contribution = attenuation * light.color *
        (max(dot(N,L),0) * albedo + material_specular_response(N,L,V))
```

This is a finite artistic falloff with peak strength near the light and zero at the range boundary;
it is explicitly not inverse-square attenuation. Gate specular by the front-facing conditions in 013.
The degeneracy check runs first, before the range check: when squared distance (Q-P)·(Q-P) is <= a
named minimum (proposed 1e-12 world-units squared), contribute zero immediately rather than normalize
an undefined direction at the exact light position — this guard sits upstream of computing `d`, `L`,
or `t`, not as a special case of the falloff formula. Only once that guard passes does the `d >= range`
branch and the `t*t` falloff apply. Use the same named degeneracy policy as 013 where applicable.
World-space numeric/transform validity retains the renderer's caller contract; robust large-world
coordinates are not introduced here.

Compute ambient once from the directional/scene ambient term. Sum directional and point contributions;
do not multiply ambient by light count. Do not tint specular by diffuse albedo. Texture color decoding
and linear-space output follow 010/013. Bright accumulated results can clamp on the current output
target; HDR/tone mapping is deferred. Each point is unshadowed even when directional shadows are enabled.

## Vulkan objects and GPU layout

Extend the existing set 2 binding 0 per-frame light UBO into one typed SceneLightUniforms block, rather
than adding a descriptor or duplicating per-frame storage for every point. Keep existing directional
fields at offsets 0/16/32 (each vec3 plus explicit padding), followed by point_count at 48 and 12 bytes
padding. The fixed point array starts at 64.

Each std140 PointLightUniforms element is 32 bytes: position vec3 at offset 0, range float at 12,
color vec3 at 16, pad float at 28. Eight elements produce a total SceneLightUniforms size of 320 bytes.
Rust mirrors use repr(C), Pod, explicit padding, and zero all padding/unused array entries. Count zero
must produce no point-light shader accesses. GLSL's fixed array bound must match MAX_POINT_LIGHTS;
inspect compiled layout/array decorations as part of verification.

Use the shared PerFrameUniform helper and 012's bounded exclusive writes. Snapshot the current
validated directional + point state into the selected frame's block only after its fence signals.
The point setter must not rewrite a mapped buffer currently read by an earlier frame. One descriptor
set bind covers the complete lighting state for that frame.

No image, sampler, or per-light allocation is created. Camera/texture/instance resources remain
unchanged. Resize does not recreate the fixed light list. Constructor rollback and teardown follow
012. No uniform-buffer-range device-limit check exists today (only `max_push_constants_size` is
tracked, in `device.rs`/`pipeline.rs`'s `validate_push_limit`); this spec adds a new one, following
that function's shape, checking the 320-byte `SceneLightUniforms` against a new
`max_uniform_buffer_range` field on `VulkanDevice` before the `Light` uniform is allocated. In
practice 320 bytes is far below Vulkan's guaranteed-minimum `maxUniformBufferRange` (16384), so this
check is a defensive floor, not an expected failure path.

## Interactions and demonstration

`light.rs` owns the public point value, validation/atomic replacement, and padded scene-light payload.
Extend the existing Light wrapper instead of creating a parallel point-only uniform implementation.
`mod.rs` stores bounded CPU light state; command.rs uploads once per submitted frame.

`triangle.frag` currently inlines directional diffuse/specular directly in `main()` (no reusable
function exists yet — see `shaders/triangle.frag`). This spec extracts a new `vec3
blinnPhongSpecular(vec3 N, vec3 L, vec3 V, vec3 specularColor, float shininess)` function (parameterized
so both the directional light and each point light call the same code, computed once per invocation
rather than copy-pasted) — this is new refactoring work, not reuse of something already factored out.
The camera-to-surface vector `V` and its degeneracy/front-facing guards are computed once per fragment
and shared across the directional call and every point-light call. Both scene shader variants from 017,
if implemented, share this same function; no duplicated shading formula or stale light layout between
variants. Recompile/commit/validate changed SPIR-V.

**`headless.rs` requires a real interface change, not just internal wiring.** `HeadlessRenderer::render_frame`
currently hardcodes `DirectionalLight::default()` with no parameter for any custom light state, and
has no way to pass point lights at all. Its signature must grow a `directional_light: &DirectionalLight`
and `point_lights: &[PointLight]` parameter alongside the existing `objects`/`camera`, matching whatever
`command::FrameContext::draw_frame`'s analogous parameters become. Without this change, spec 018's
entire GPU acceptance test plan (below) is unimplementable, since there is currently no way to render a
scene with a non-default light configuration headlessly.

Cube example: two differently colored positioned lights with overlapping finite ranges, a stationary
or disabled directional diffuse term, and the existing movable camera. Move one light along a known
path to show dynamic updates; display its position/range through the existing text API if useful.
Keep the library free of example animation/input ownership.

## Test plan

CPU: valid empty/one/eight lists, rejection of nine, invalid components in each field, range boundaries,
zero/HDR colors, and unchanged prior list after any invalid element. Check 320-byte block size,
32-byte array stride, exact offsets, count, zero padding/unused entries, and default zero count. Check
uniform-range/device-limit handling and fixed-buffer bounds. Test real production payload construction,
not an independent approximation.

GPU shader acceptance uses spec 019's headless/golden-image infrastructure (see Resolved decisions):
zero points matches the 013 baseline; one light illuminates nearby surfaces; colored lights add;
beyond range contributes zero; near-boundary falloff is smooth; no invalid output at a light position;
specular follows camera position; ambient does not grow with point count. Use `HeadlessRenderer` with
known geometry/light positions and `assert_matches_golden` against committed reference PNGs at
spec 019's default tolerance (2/255 per channel). Test dynamic per-frame updates with two frames in
flight, all eight lights, text, empty scenes, resize and clean teardown under synchronization
validation (manual run, same as prior specs — spec 019 did not build resize/text coverage). With 017
(not yet implemented), directional visibility must not incorrectly darken point-light contributions —
deferred until 017 lands. Do not claim CPU reference math proves the GLSL result; each golden-image
test reads the actual shader's rendered output.

Run all quality gates/doctests and shader compilation/validation. GPU golden-image tests are
`#[ignore]`d by default (consistent with spec 019) and run locally via `cargo test -- --ignored`; a
machine with a Vulkan driver must generate each golden PNG once with `M2S2_UPDATE_GOLDEN=1` before the
comparison tests can pass, same workflow spec 019 documented in `CLAUDE.md`.

## Resolved decisions (2026-11-08)

- **Bounded forward-light model:** accepted as proposed — up to `MAX_POINT_LIGHTS = 8` lights, squared
  artistic falloff (`t = 1 - d/range; attenuation = t*t`), zero contribution at/beyond `range`. Not
  physically based inverse-square; this is an intentional, bounded, finite-range model matching the
  spec's stated Non-goals.
- **Degenerate light position:** accepted as proposed — when squared distance is below the named
  minimum (1e-12, matching 013's degeneracy policy), contribute zero rather than defining an arbitrary
  light direction. Consistent with 013's "never normalize a zero vector" rule.
- **Shared shader organization:** accepted as proposed — the point-light response reuses 013's existing
  Blinn–Phong diffuse/specular functions in `triangle.frag`, called once per active light with that
  light's `L`, and summed; no duplicated shading formula. 017's shadow visibility (when implemented)
  gates only the directional term, per this spec's Dependencies note — point lights stay unaffected
  until a point-light shadow spec is written (explicitly a Non-goal here).
- **GPU verification method:** **changed from the spec's original manual-visual-check assumption.**
  Spec 019 (headless rendering + golden-image comparison) is now implemented and merged — built
  specifically so specs like this one could get real automated pixel-comparison acceptance tests
  instead of a human eyeballing the cube example. This spec's GPU acceptance tests use
  `HeadlessRenderer` + `assert_matches_golden` at spec 019's default tolerance (2/255 per channel per
  channel), covering the scenarios listed in the Test plan's GPU section above. Golden PNGs are
  generated once by the user (same `M2S2_UPDATE_GOLDEN=1` handoff spec 019 established) and committed
  under `testdata/golden/`. Resize/text/empty-scene/synchronization-validation checks remain a manual
  run on the cube example, since spec 019 did not build automated coverage for those paths.

## Open questions for review

None blocking. Everything above was a design-blocking decision; all are resolved.

## Outcome

Pending implementation.
