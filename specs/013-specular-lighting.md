# 013: Blinn–Phong specular lighting

**Status**: Draft for review; not implemented.
**Roadmap item**: [Phase 2 basic lighting](../ROADMAP.md#phase-2-standard-rendering-features).
**Prerequisites**: Spec 011 directional lighting and the ownership/mapped-write contracts agreed in
spec 012. This draft can be reviewed before 012 is implemented; it must not bypass its safety work.

## Goal

Add camera-dependent Blinn–Phong highlights to the existing directional light, with per-object
specular color and shininess. Demonstrate how diffuse surface response differs from a reflected
highlight while preserving the current texture, vertex color, ambient, and depth paths.

## Non-goals

PBR/energy conservation, a material resource registry, point/spot or multiple lights, shadows, normal
maps, specular textures, transparency, HDR render targets, tone mapping, and new mesh/resource ownership.
This model is an educational extension of 011, not a physically based material system.

## Public API: proposed pre-1.0 migration

Replace separate render-call view/projection arguments with one explicit frame camera value. The
caller supplies the world-space position used to construct its view matrix; do not add a persistent
camera-position setter that can silently become stale.

```rust
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3, // world space
}

#[derive(Copy, Clone, Debug)]
pub struct SpecularMaterial {
    pub color: Vec3,   // linear RGB reflection strength; finite, nonnegative
    pub shininess: f32, // finite, >= 1
}

impl Default for SpecularMaterial {
    // Proposed: color (0,0,0), shininess 32; existing diffuse-only output is the default.
    fn default() -> Self;
}

pub struct Object {
    pub model: Mat4,
    pub texture: Option<TextureId>,
    pub specular: SpecularMaterial,
}

impl VulkanRenderer {
    pub fn render_frame(&mut self, model: Mat4, camera: Camera) -> Result<()>;
    pub fn render_objects(&mut self, objects: &[Object], camera: Camera) -> Result<()>;
    pub fn render_objects_with_text(
        &mut self, objects: &[Object], camera: Camera, text: &[TextDraw<'_>],
    ) -> Result<()>;
}
```

Re-export Camera and SpecularMaterial. This intentionally changes existing render signatures and
Object literals; migrate both examples and README together. `render_frame` supplies the default
specular material. Do not invent a camera position for legacy view/proj-only calls. Matrix validity,
invertible model transforms, and meaningful vertex normals retain the existing caller contract.
Validate camera position and every object's specular values before acquire, mapped writes, or fence
reset. Reject non-finite position, negative/non-finite color, or shininess below one/non-finite with
ValidationError. Color values above one are allowed as explicit intensities. Invalid input must not
partially submit objects. A zero specular color disables highlights without skipping validation.

The directional-light API and its validation remain unchanged. Specular reflection uses the existing
light color; there is no second light resource or separate light-intensity setting.

## Coordinate spaces and shading

All lighting vectors use world space:
- Vertex shader outputs world position `(model * vec4(position, 1)).xyz` and transformed normal.
- Use the inverse-transpose model normal matrix. Interpolate its result and normalize in the fragment
  shader; avoid normalizing before interpolation when supporting varying normals/non-uniform scale.
- N is the surface normal, L is normalized negative light travel direction, V points from surface to
  camera, and H is normalized L + V.

The intended fragment RGB is:

```text
albedo = texture.rgb * vertexColor
ndotl = max(dot(N, L), 0)
diffuseAmbient = (light.ambient + light.color * ndotl) * albedo
highlight = light.color * material.color * pow(max(dot(N, H), 0), material.shininess)
result.rgb = diffuseAmbient + highlight
result.a = texture.a
```

Evaluate highlight only when ndotl > 0, dot(N,V) > 0, and V and L+V have safe nonzero lengths. Otherwise
highlight is zero. Use a named minimum squared-vector length (proposed 1e-12) for these degeneracies;
never normalize a zero view or halfway vector. Do not tint the highlight by diffuse albedo or vertex
color. Work in linear RGB; existing sRGB texture reads and swapchain encoding remain responsible for
conversion. Bright results may clamp on the current output target; tone mapping is deferred.

## Vulkan objects and lifetime

No new per-object VkBuffer or descriptor sets. Extend the camera's existing per-frame UBO to include
position and padding: view at offset 0, proj at 64, position at 128, pad at 140; total 144 bytes.
Set 0 binding 0 becomes visible in VERTEX | FRAGMENT. GLSL keeps row_major declarations and matching
std140 layout. Camera updates continue using spec 012's checked exclusive per-frame writes after the
slot's fence wait.

Extend the per-draw push block: model at offset 0 (64 bytes), specular color at 64 (12 bytes), shininess
at 76 (4 bytes), total 80 bytes. Declare one offset-zero range with VERTEX | FRAGMENT visibility; record
all bytes before each indexed draw. Both shader block declarations must agree on layout and pipeline
creation must validate the complete range against maxPushConstantsSize. Use named offsets/stages and
derived Rust sizes. Keep texture at set 1 and directional light at set 2.

Rebuild scene pipeline/shaders with the changed camera descriptor visibility and push range. Text
uses its own pipeline/layout and receives no scene material data. Destruction order and allocation
ownership remain as established by 012. Update tests/tools to the new layouts rather than retaining
parallel old/new uniform paths.

## Interactions and demonstration

- `uniform.rs`: Camera payload, conversion/padding, both-stage visibility, size/offset tests.
- `pipeline.rs`: material push payload/range and limit check.
- `command.rs`: push per-object material bytes alongside model; camera update remains once per frame.
- `texture.rs` (Object's current namespace): add the material field; keep TextureId unchanged.
- `mod.rs`/`lib.rs`: frame camera API and early input validation.
- Triangle shaders: world position varying, camera position, and Blinn–Phong term; rebuild and commit
  both SPIR-V files with glslc and validate them with spirv-val.
- Cube example: enable highlights explicitly, using identical specular color and different shininess
  (proposed 16 and 64). Camera motion changes highlights while the world-space sun stays stationary.
  Include a specular-off comparison using zero material color. Keep controls and text working.
- Triangle example: migrate to explicit Camera and default material behavior.

## Test plan

CPU: camera/push sizes, offsets, and zeroed padding; shader parse and SPIR-V member-offset inspection;
range size/stage/limit boundaries; valid defaults, zero color, HDR color, invalid color/shininess/position;
zero/one/two objects record the right material bytes in the existing production emission test seam.
Rejected frames issue no submission and leave the next valid frame usable.

Test shader math through GPU output or a shared compiled shader test path. Do not claim that a separate
Rust implementation proves the GLSL. CPU reference calculations may provide expected pixel values,
but the acceptance comparison must read the actual shader's result.

GPU: fixed white geometry/material isolates lighting from albedo. Compare specular off/on, broad/sharp
highlights, camera movement under a stationary sun, back-facing/unlit surfaces, degenerate view/halfway
vectors, and non-uniform scale. Confirm the specular-off result matches the current diffuse baseline
within an explicit image-comparison tolerance. Verify texture/overlay, empty frames, resize and teardown
with synchronization validation. Record the chosen pixel tolerance and tested device in Outcome.

Run all repository quality gates, doctests, shader compilation/validation, and documented API examples.
Keep GPU checks opt-in for CPU-only CI.

## Open questions for review

- Accept the explicit Camera signature migration, or define a named alternate API before implementation?
- Accept zero specular color as the compatibility default, with the cube opting into visible highlights?
- Agree on the GPU image-comparison tolerance and test fixture before accepting shader results.

## Outcome

Pending review and implementation. No rendering code is changed by this draft.

## References

- [Khronos push-constant layout and stage compatibility](https://docs.vulkan.org/guide/latest/push_constants.html)
- Existing specs 007, 011, and draft 012 for model data, lighting spaces, and resource safety.
