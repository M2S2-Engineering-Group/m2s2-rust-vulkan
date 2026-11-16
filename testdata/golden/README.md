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
