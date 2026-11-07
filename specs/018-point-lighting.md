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
When squared distance is <= a named minimum (proposed 1e-12 world-units squared), contribute zero rather
than normalize an undefined direction at the exact light position. Use the same named degeneracy
policy as 013 where applicable. World-space numeric/transform validity retains the renderer's caller
contract; robust large-world coordinates are not introduced here.

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
012; the larger light buffer must satisfy the device's uniform-range limit before allocation.

## Interactions and demonstration

`light.rs` owns the public point value, validation/atomic replacement, and padded scene-light payload.
Extend the existing Light wrapper instead of creating a parallel point-only uniform implementation.
`mod.rs` stores bounded CPU light state; command.rs uploads once per submitted frame. Both scene shader
variants from 017, if implemented, share directional/point response functions; no duplicated shading
formula or stale light layout between variants. Recompile/commit/validate changed SPIR-V.

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

GPU shader acceptance: zero points matches the 013 baseline; one light illuminates nearby surfaces;
colored lights add; beyond range contributes zero; near-boundary falloff is smooth; no invalid output
at a light position; specular follows camera position; ambient does not grow with point count. Use
known geometry/light positions and read actual pixels with agreed tolerance. Test dynamic per-frame
updates with two frames in flight, all eight lights, text, empty scenes, resize and clean teardown under
synchronization validation. With 017, directional visibility must not incorrectly darken point-light
contributions. Do not claim CPU reference math proves the GLSL result.

Run all quality gates/doctests and shader compilation/validation. GPU tests stay opt-in for CPU CI.

## Open questions for review

- Accept eight lights and a squared finite-range falloff as the initial bounded forward-light model?
- Accept zero contribution at the exact light position instead of defining an arbitrary light direction?
- Confirm the shared shader organization and numerical comparison tolerance after 013 review.

## Outcome

Pending design review and implementation. Spot lights and point-light shadows remain future specs.
