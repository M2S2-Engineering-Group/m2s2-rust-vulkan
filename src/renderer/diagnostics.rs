//! CPU-only diagnostic formatting. Its output can be submitted as any other text.
use crate::math::Vec3;
use ash::vk;
use std::{collections::VecDeque, time::Duration};

const TIMING_WINDOW: usize = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderStatus {
    Ready,
    ResizePending,
    Paused,
}

/// Vulkan state plus counts from the most recent submitted scene frame (overlay excluded).
pub struct RendererStats<'a> {
    pub device_name: &'a str,
    pub extent: (u32, u32),
    pub color_format: vk::Format,
    pub image_count: usize,
    pub scene_draw_calls: usize,
    pub submitted_indices: u64,
    pub status: RenderStatus,
}

#[derive(Default)]
pub struct Diagnostics {
    samples: VecDeque<Duration>,
}

impl Diagnostics {
    /// Add caller-measured CPU frame cadence, including any CPU waits. Zero samples are ignored.
    pub fn update(&mut self, frame_interval: Duration) {
        if frame_interval.is_zero() {
            return;
        }
        if self.samples.len() == TIMING_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(frame_interval);
    }

    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Average FPS and milliseconds over the last 60 positive samples. No GPU timing is inferred.
    pub fn timing(&self) -> Option<(f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let mean =
            self.samples.iter().map(Duration::as_secs_f64).sum::<f64>() / self.samples.len() as f64;
        Some((1.0 / mean, mean * 1000.0))
    }

    pub fn format(&self, stats: &RendererStats<'_>, camera_position: Vec3) -> String {
        let timing = self.timing().map_or_else(
            || "FPS: N/A | CPU frame: N/A".into(),
            |(fps, ms)| format!("FPS: {fps:.1} | CPU frame: {ms:.2} ms"),
        );
        format!("Camera: ({:.2}, {:.2}, {:.2})\n{}\nGPU: {}\nSwapchain: {}x{} | {:?} | {} images\nScene: {} draws | {} indices\nState: {:?}",
            camera_position[0], camera_position[1], camera_position[2], timing, stats.device_name,
            stats.extent.0, stats.extent.1, stats.color_format, stats.image_count,
            stats.scene_draw_calls, stats.submitted_indices, stats.status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timing_ignores_zero_and_reset_restores_unavailable() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.update(Duration::ZERO);
        assert_eq!(diagnostics.timing(), None);
        diagnostics.update(Duration::from_millis(10));
        diagnostics.update(Duration::from_millis(30));
        assert_eq!(diagnostics.timing(), Some((50.0, 20.0)));
        diagnostics.reset();
        assert_eq!(diagnostics.timing(), None);
    }
    #[test]
    fn timing_window_evicts_oldest() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.update(Duration::from_secs(1));
        for _ in 0..TIMING_WINDOW {
            diagnostics.update(Duration::from_millis(10));
        }
        let (fps, ms) = diagnostics.timing().unwrap();
        assert!((fps - 100.0).abs() < 0.00001);
        assert!((ms - 10.0).abs() < 0.00001);
        assert_eq!(diagnostics.samples.len(), TIMING_WINDOW);
    }
    #[test]
    fn formats_explicit_camera_metrics_and_unavailable_timing() {
        let stats = RendererStats {
            device_name: "Test GPU",
            extent: (800, 600),
            color_format: vk::Format::R8G8B8A8_UNORM,
            image_count: 3,
            scene_draw_calls: 2,
            submitted_indices: 72,
            status: RenderStatus::ResizePending,
        };
        let mut diagnostics = Diagnostics::default();
        let text = diagnostics.format(&stats, Vec3::new(1.0, -2.0, 3.0));
        for expected in [
            "Camera: (1.00, -2.00, 3.00)",
            "FPS: N/A",
            "Test GPU",
            "800x600",
            "3 images",
            "2 draws | 72 indices",
            "ResizePending",
        ] {
            assert!(text.contains(expected));
        }
        diagnostics.update(Duration::from_millis(20));
        assert!(diagnostics
            .format(&stats, Vec3::new(0.0, 0.0, 0.0))
            .contains("FPS: 50.0 | CPU frame: 20.00 ms"));
    }
}
