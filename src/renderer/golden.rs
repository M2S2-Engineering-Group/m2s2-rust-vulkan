/// Golden-image PNG comparison with per-channel tolerance for testing.
pub(crate) fn assert_matches_golden(
    actual: &[u8],
    width: u32,
    height: u32,
    path: &std::path::Path,
    tolerance: u8,
) -> std::result::Result<(), String> {
    // If M2S2_UPDATE_GOLDEN is set, write the actual image as the golden and return Ok.
    if std::env::var("M2S2_UPDATE_GOLDEN").is_ok() {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("failed to create golden image at {path:?}: {e}"))?;
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgba);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("failed to write PNG header: {e}"))?;
        writer
            .write_image_data(actual)
            .map_err(|e| format!("failed to write PNG image data: {e}"))?;
        return Ok(());
    }

    // Open and decode the golden image.
    let file = std::fs::File::open(path).map_err(|_| {
        format!(
            "no golden image at {path:?} (or it failed to decode): run with M2S2_UPDATE_GOLDEN=1 to create it"
        )
    })?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("failed to decode PNG: {e}"))?;
    let mut golden_pixels = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut golden_pixels)
        .map_err(|e| format!("failed to read PNG frame: {e}"))?;
    let golden_info = reader.info();

    // Check dimensions match.
    if golden_info.width != width || golden_info.height != height {
        return Err(format!(
            "golden image dimension mismatch: golden is {}x{}, actual is {}x{}",
            golden_info.width, golden_info.height, width, height
        ));
    }

    // Compare pixels within tolerance.
    let pixels_count = (width * height) as usize;
    for i in 0..pixels_count {
        let golden_px = [
            golden_pixels[i * 4],
            golden_pixels[i * 4 + 1],
            golden_pixels[i * 4 + 2],
            golden_pixels[i * 4 + 3],
        ];
        let actual_px = [
            actual[i * 4],
            actual[i * 4 + 1],
            actual[i * 4 + 2],
            actual[i * 4 + 3],
        ];

        for (golden_ch, actual_ch) in golden_px.iter().zip(actual_px.iter()) {
            let diff = (*golden_ch as i16 - *actual_ch as i16).unsigned_abs() as u8;
            if diff > tolerance {
                let x = (i as u32) % width;
                let y = (i as u32) / width;
                return Err(format!(
                    "pixel ({}, {}) differs: golden {:?} vs actual {:?} (tolerance {})",
                    x, y, golden_px, actual_px, tolerance
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_png(path: &std::path::Path, pixels: &[u8], width: u32, height: u32) {
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgba);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(pixels).unwrap();
    }

    #[test]
    fn identical_images_pass_at_zero_tolerance() {
        let path = std::env::temp_dir().join("m2s2_golden_identical.png");
        let pixels = vec![10u8, 20, 30, 255, 40, 50, 60, 255]; // 2x1 RGBA
        write_png(&path, &pixels, 2, 1);
        assert!(assert_matches_golden(&pixels, 2, 1, &path, 0).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn out_of_tolerance_pixel_fails_with_coordinates() {
        let path = std::env::temp_dir().join("m2s2_golden_mismatch.png");
        let golden = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        write_png(&path, &golden, 2, 1);
        let mut actual = golden.clone();
        actual[4] = 200; // second pixel's red channel, far outside any reasonable tolerance
        let err = assert_matches_golden(&actual, 2, 1, &path, 2).unwrap_err();
        assert!(
            err.contains("1, 0") || err.contains("(1,0)"),
            "error should name the differing pixel's coordinates: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn within_tolerance_difference_passes() {
        let path = std::env::temp_dir().join("m2s2_golden_tolerance.png");
        let golden = vec![10u8, 20, 30, 255];
        write_png(&path, &golden, 1, 1);
        let actual = vec![11u8, 19, 31, 255]; // every channel within 2
        assert!(assert_matches_golden(&actual, 1, 1, &path, 2).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_golden_without_update_flag_fails_clearly() {
        let path = std::env::temp_dir().join("m2s2_golden_missing_no_update.png");
        let _ = std::fs::remove_file(&path);
        std::env::remove_var("M2S2_UPDATE_GOLDEN");
        let err = assert_matches_golden(&[0u8, 0, 0, 255], 1, 1, &path, 2).unwrap_err();
        assert!(
            err.contains("M2S2_UPDATE_GOLDEN"),
            "error should mention the update flag: {err}"
        );
    }

    #[test]
    fn missing_golden_with_update_flag_writes_one() {
        let path = std::env::temp_dir().join("m2s2_golden_missing_with_update.png");
        let _ = std::fs::remove_file(&path);
        std::env::set_var("M2S2_UPDATE_GOLDEN", "1");
        let result = assert_matches_golden(&[1u8, 2, 3, 255], 1, 1, &path, 2);
        std::env::remove_var("M2S2_UPDATE_GOLDEN");
        assert!(result.is_ok());
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dimension_mismatch_fails_clearly_not_panics() {
        let path = std::env::temp_dir().join("m2s2_golden_dimension_mismatch.png");
        write_png(&path, &[0u8, 0, 0, 255, 0, 0, 0, 255], 2, 1);
        let err = assert_matches_golden(&[0u8, 0, 0, 255], 1, 1, &path, 2).unwrap_err();
        assert!(
            err.to_lowercase().contains("dimension") || err.contains("2"),
            "error should mention the dimension mismatch: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }
}
