pub mod error;
pub mod math;
pub mod renderer;

pub use error::{Result, VulkanError};
pub use renderer::VulkanRenderer;

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
