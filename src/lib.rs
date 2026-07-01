pub mod renderer;
pub mod math;
pub mod error;

pub use renderer::VulkanRenderer;
pub use error::{VulkanError, Result};

/// Initialize logging for the library
pub fn init_logging() {
    env_logger::init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
