# testdata/golden

Golden PNGs for the `#[ignore]`d GPU golden-image tests in `src/renderer/headless.rs` (e.g.
`headless_self_test.png` for `self_test_scene_matches_golden_with_zero_validation_errors`) live here.
They are generated, not hand-authored — see the "Headless GPU tests and golden images" section of
`CLAUDE.md` for how to generate or regenerate one.


Spec 015 adds three independently inspected baselines:

- `multi_mesh_different.png`: a cube and a separate triangle mesh.
- `multi_mesh_repeated.png`: two objects sharing the same cube handle.
- `multi_mesh_text_only.png`: text with zero meshes uploaded.

Regenerate only these scenarios deliberately on a machine with a Vulkan driver:

```sh
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless::tests::multi_mesh -- --ignored --test-threads=1
```

Then compare all GPU tests against the stored images without the update flag:

```sh
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT cargo test --lib -- --ignored --test-threads=1
```

The multi-mesh rejection test has no golden file; it checks invalid handles/uploads and successful
rendering afterward. Never regenerate existing baselines to hide a rendering regression.

Spec 017 adds `shadow_disabled.png` and `shadow_cube_on_plane.png`: the same textured cube and
separate receiver plane with shadows disabled/enabled. Generated and inspected on Apple M5 /
MoltenVK 1.4.1 using D32_SFLOAT, 256-square shadow maps, bias 1.25/1.75, and 96-square output;
comparison tolerance is 2 per channel. The depth map itself is independently tested by GPU readback.

Regenerate only these two references deliberately:

```sh
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless::tests::shadow_cube_on_plane_matches_goldens -- --ignored --test-threads=1
```

Then rerun the full ignored suite without the update flag using the synchronization-validation
command above. Settings and remaining interactive verification limits are recorded in spec 017.

Spec 021 adds `material_textured_tint.png` and `material_untextured_tint.png`: the same lit cube
and camera as the self-test baseline, but with `Material.base_color` set to a non-white tint
`(0.3, 0.6, 1.0)` — one with an explicit 1x1 white texture bound, one with no texture at all —
proving the tint reaches the shader in both the textured and untextured paths.

Regenerate only these two references deliberately:

```sh
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless::tests::material_textured_tint_matches_golden -- --ignored --test-threads=1
M2S2_UPDATE_GOLDEN=1 cargo test --lib renderer::headless::tests::material_untextured_tint_matches_golden -- --ignored --test-threads=1
```

Then rerun the full ignored suite without the update flag using the synchronization-validation
command above.
