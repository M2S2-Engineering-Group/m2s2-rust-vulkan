use std::fmt;

pub type Result<T> = std::result::Result<T, VulkanError>;

#[derive(Debug)]
pub enum VulkanError {
    VulkanError(ash::vk::Result),
    WindowError(String),
    InitializationError(String),
    ValidationError(String),
    Other(String),
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanError::VulkanError(result) => write!(f, "Vulkan error: {:?}", result),
            VulkanError::WindowError(msg) => write!(f, "Window error: {}", msg),
            VulkanError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
            VulkanError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            VulkanError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for VulkanError {}

impl From<ash::vk::Result> for VulkanError {
    fn from(result: ash::vk::Result) -> Self {
        VulkanError::VulkanError(result)
    }
}
