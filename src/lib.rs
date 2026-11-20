pub mod error;
pub mod math;
mod renderer;

pub use error::{Result, VulkanError};
pub use renderer::buffer::{Index, Vertex};
pub use renderer::diagnostics::{Diagnostics, RenderStatus, RendererStats};
pub use renderer::light::{DirectionalLight, PointLight, MAX_POINT_LIGHTS};
pub use renderer::mesh::MeshId;
pub use renderer::scene::Object;
pub use renderer::shadow::DirectionalShadow;
pub use renderer::text::TextDraw;
pub use renderer::texture::{Mipmaps, SpecularMaterial, TextureId, TextureOptions};
pub use renderer::uniform::Camera;
pub use renderer::VulkanRenderer;
// Re-exported so consumers can name the handle types `VulkanRenderer::new` takes without
// depending on `raw-window-handle` directly.
pub use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// Initialize logging for the library
pub fn init_logging() {
    env_logger::init();
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
