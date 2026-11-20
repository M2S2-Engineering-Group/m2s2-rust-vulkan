# 021: Material system

**Status**: Implemented; unit tests and real-hardware GPU golden-image acceptance verified on
Apple M5 / MoltenVK (2026-11-20).
**Roadmap item**: Phase 3 "Material system: decouple 'what a shader needs' from 'what an asset
provides'".
**Dependencies**: Spec 010's texture creation/`TextureId`; spec 013's `SpecularMaterial`/Blinn-Phong
shading; spec 016's `InstanceData` per-instance vertex attributes; spec 017's `SHADOWS_ENABLED`
shader-macro structure and `shadow_depth.vert`'s subset-of-attributes precedent (this spec's shader
changes build directly on both); spec 020's `LoadedMesh`/glTF loading. All implemented and merged.

## Goal

Introduce one `Material` type — texture, a base-color tint, and the existing specular
color/shininess — that replaces `Object`'s separate `texture`/`specular` fields, and extend
`src/model.rs`'s glTF loader to surface `baseColorFactor` into it. Today, spec 020's loader only
ever captures a mesh's base-color *texture* reference; a glTF material's flat tint color (common in
real-world assets, frequently used with no texture at all) has nowhere to go and is silently
dropped, so such meshes render plain white regardless of their authored color.

## Non-goals

- Metallic/roughness BRDF, image-based lighting, emissive/normal/occlusion textures, alpha
  blending or cutout, double-sided culling — all a separate, later "PBR materials" ROADMAP item.
  This spec changes no shading math beyond one multiplicative tint; the lighting model stays
  exactly the Blinn-Phong spec 013 shipped.
- Capturing metallic/roughness factors as inert, unused data "for later" — nothing in this engine
  consumes them yet, and speculative plumbing for values nothing uses is explicitly out of scope.
- A `MaterialId`/renderer-owned material store (mirroring `MeshId`/`TextureId`). `Material` stays a
  plain `Copy` value type on `Object`, same lifetime/ownership model as today's `SpecularMaterial`.
- glTF material `name`, extensions, or multiple materials per primitive (spec 020 already rejects
  multi-primitive meshes).
- Any change to `examples/cube.rs`'s or `examples/model.rs`'s visual output — both must look
  pixel-identical after migrating to `Material`, since every default preserves current behavior.

## Public API

```rust
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub texture: Option<TextureId>,
    pub base_color: Vec3,
    pub specular: SpecularMaterial,
}

impl Default for Material {
    // texture: None, base_color: (1,1,1) (no tint), specular: SpecularMaterial::default() —
    // identical visual result to today's `Object { texture: None, specular: default, .. }`.
    fn default() -> Self;
}

pub struct Object {
    pub model: Mat4,
    pub mesh: MeshId,
    pub material: Material,   // replaces the separate `texture`/`specular` fields
}
```

`src/model.rs`:
```rust
pub struct LoadedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
    pub world_transform: Mat4,
    pub texture: Option<usize>,
    pub base_color_factor: Vec3,   // new: glTF baseColorFactor, alpha dropped (no blending support)
}
```
`base_color_factor` always has a value — glTF's `PbrMetallicRoughness::base_color_factor()`
defaults to `[1.0, 1.0, 1.0, 1.0]` when the material/document doesn't set it, and when a primitive
has no material at all (already a supported case per spec 020 — `texture: None`), this spec's
`base_color_factor` is likewise `(1.0, 1.0, 1.0)`. Callers build `Material { texture:
texture.map(|i| texture_ids[i]), base_color: loaded_mesh.base_color_factor, specular: /* caller's
choice, unchanged from today */ }`.

## Vulkan objects

`InstanceData` (`pipeline.rs`) gains a sixth `vec4` field, `base_color` (`xyz` = tint, `w` unused/
padding — same packing convention spec 016 already established for `specular`), occupying a new
per-instance vertex-input attribute at location 9. Only the main scene pipeline's shaders read it;
`shadow_depth.vert` needs no change at all, since a Vulkan vertex shader may already declare a
subset of the bound vertex-input attributes (precedent: it already omits location 8's `specular`).

## Interactions

- `validate_specular` (unchanged) is called on `object.material.specular`. A new
  `validate_material` wraps it and additionally rejects a non-finite or negative `base_color`
  component, mirroring the existing specular-color check.
- `resolve_texture_set` is called with `object.material.texture` (unchanged signature).
- `command.rs`'s per-frame `InstanceData::from(...)` conversion gains `base_color` as a new tuple
  element, sourced from `object.material.base_color`; `mod.rs`'s per-object validation loop calls
  `validate_material(&object.material)` instead of `validate_specular(&object.specular)`.
- `triangle.vert`/`triangle.frag` (both the plain and `-DSHADOWS_ENABLED` variants — recompile all
  four `.spv` outputs per `CLAUDE.md`'s shader-recompilation table) gain a new unconditional
  `fragBaseColor` varying at location 5; `SHADOWS_ENABLED`'s `fragLightSpacePos` shifts from
  location 5 to 6 to stay after it (both shader stages of each variant updated in lockstep — this
  is an internal detail of one self-consistent compiled pair, not a public API). The fragment
  shader's existing `albedo` computation gains one multiply: `texture(tex, fragUV) *
  vec4(fragColor, 1.0) * vec4(fragBaseColor.rgb, 1.0)`.
- `examples/cube.rs` and `examples/model.rs` both migrate their `Object` construction from
  `{ texture, specular }` to `{ material: Material { texture, base_color: Vec3::new(1.0,1.0,1.0),
  specular } }` (or, for `model.rs`, `base_color: loaded_mesh.base_color_factor`) — no visual
  change for `cube.rs` (neutral tint); `model.rs` is the first real consumer of a non-trivial
  `base_color_factor`, though the converted Mecha-Ziu asset's own material happens to use the
  glTF default `(1,1,1)` tint, so no visual change is expected there either — this spec adds the
  *capability*, not necessarily a visible difference in the one asset already committed.

## Test plan

- Unit tests (no GPU): `Material::default()` matches today's `Object` defaults exactly;
  `validate_material` rejects non-finite/negative `base_color` components and delegates specular
  validation unchanged; `InstanceData`'s new `base_color` field packs at the expected byte offset
  and the attribute-description list reports 6 locations (4 through 9) instead of 5; `src/model.rs`
  tests for `base_color_factor`: explicit non-default glTF `baseColorFactor`, glTF default (field
  omitted), and no material at all — all three read back as the expected `Vec3`.
- Headless GPU acceptance (per spec 019's infra): two new *committed* reference PNGs under
  `testdata/golden/` (same `assert_matches_golden`/`M2S2_UPDATE_GOLDEN=1` workflow every existing
  golden already uses, not an in-test frame-to-frame diff) — one textured, tinted object (proves
  the tint actually reaches the fragment shader), one *untextured* tinted object (proves it no
  longer renders plain white — the concrete gap this spec closes). The existing
  `self_test_scene_matches_golden_with_zero_validation_errors` baseline (neutral `(1,1,1)` tint
  throughout, via `Material::default()`) already proves the multiply is an identity at default,
  since that golden must remain byte-identical after this spec's `Object`/`InstanceData` shape
  change — no separate "tinted vs. neutral" comparison golden is needed. Confirm all pre-existing
  goldens remain byte-identical (zero regression from the shape change alone).

## Open questions

- Exact `InstanceData` padding choice for `base_color` (a true `vec3` attribute vs. a `vec4` with
  an unused `w`) — resolve during implementation; the existing `specular` field's `vec4` packing
  (`xyz`/`w` = color/shininess) is a reasonable precedent to match for consistency, even though
  `base_color`'s `w` has no corresponding scalar to carry.
- Whether to give `examples/cube.rs` a cosmetic non-neutral tint on one cube to visually demonstrate
  the feature, or keep both at the neutral default to avoid an unrelated example-visual change —
  resolve during implementation.

## Outcome

Implemented as specified. `Material { texture: Option<TextureId>, base_color: Vec3, specular:
SpecularMaterial }` replaces `Object`'s separate `texture`/`specular` fields, with a
`validate_material` that delegates specular validation and additionally rejects a non-finite or
negative `base_color` component. `src/model.rs`'s `LoadedMesh` gained `base_color_factor: Vec3`,
read from `pbr_metallic_roughness().base_color_factor()` — the same accessor call already handles
"explicit factor", "material present but factor omitted" (glTF's own `[1,1,1,1]` default), and "no
material at all" identically, so no extra branching was needed. `examples/model.rs` wires
`loaded_mesh.base_color_factor` straight into each `Object`'s `Material`, closing the real gap this
spec targeted: an untextured, tinted glTF material no longer silently renders plain white.

`InstanceData` (`pipeline.rs`) gained a sixth per-instance `vec4` (`base_color`, `xyz` = tint, `w`
unused padding — same convention as `specular`) at vertex-input location 9, growing the struct from
80 to 96 bytes. `triangle.vert`/`triangle.frag` (both the plain and `-DSHADOWS_ENABLED` variants, all
four `.spv` outputs recompiled) gained an unconditional `fragBaseColor` varying at location 5, with
`SHADOWS_ENABLED`'s `fragLightSpacePos` shifted to location 6; the fragment shader's albedo
computation gained one multiply by `fragBaseColor.rgb`. `shadow_depth.vert` needed no change, per
the spec's own precedent (a vertex shader may bind a subset of the available per-instance
attributes).

Both open questions resolved during implementation: `base_color` uses a `vec4` (matching
`specular`'s packing for consistency, `w` genuinely unused), and `examples/cube.rs` kept both cubes
at the neutral default tint — no cosmetic demonstration was added, since the spec's own non-goal
("no visual change to `cube.rs`") made that the simpler, lower-risk choice.

Two new committed goldens (`material_textured_tint.png`, `material_untextured_tint.png`) prove the
tint reaches the shader in both the textured and untextured paths. Confirmed on real hardware (Apple
M5): every pre-existing golden PNG remained byte-identical after the `Object`/`InstanceData` shape
change (`git diff --stat testdata/golden/` showed only the two new files, zero existing-file diffs),
and the full ignored GPU suite (34 tests) passed with
`VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` enabled. A test
(`batched_objects_keep_their_own_distinct_base_color`, added to this
spec's own plan during self-review) confirms the spec's batching-distinctness requirement: two
objects sharing mesh and texture still batch into one instanced draw call, and each instance keeps
its own `base_color` rather than collapsing to a shared value.
