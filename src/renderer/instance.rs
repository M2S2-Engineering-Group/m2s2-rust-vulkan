use ash::{vk, Entry, Instance};
use std::ffi::{CStr, CString};
use crate::error::{Result, VulkanError};

pub struct VulkanInstance {
    pub entry: Entry,
    pub instance: Instance,
    #[cfg(debug_assertions)]
    pub debug_utils: Option<ash::ext::debug_utils::Instance>,
    #[cfg(debug_assertions)]
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanInstance {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| VulkanError::InitializationError(format!("Failed to load Vulkan: {}", e)))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(CStr::from_bytes_with_nul(b"M2S2 Vulkan Renderer\0").unwrap())
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(CStr::from_bytes_with_nul(b"M2S2 Engine\0").unwrap())
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);

        let mut extension_names = vec![
            ash::khr::surface::NAME.as_ptr(),
        ];

        // Platform-specific surface extensions
        #[cfg(target_os = "windows")]
        extension_names.push(ash::khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "macos")]
        extension_names.push(ash::ext::metal_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        extension_names.push(ash::khr::xlib_surface::NAME.as_ptr());

        let layer_names = vec![];

        #[cfg(debug_assertions)]
        {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(VulkanError::from)?;

        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) = Self::setup_debug_messenger(&entry, &instance)?;

        Ok(Self {
            entry,
            instance,
            #[cfg(debug_assertions)]
            debug_utils: Some(debug_utils),
            #[cfg(debug_assertions)]
            debug_messenger: Some(debug_messenger),
        })
    }

    #[cfg(debug_assertions)]
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        let debug_utils = ash::ext::debug_utils::Instance::new(entry, instance);

        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        let debug_messenger = unsafe {
            debug_utils.create_debug_utils_messenger(&create_info, None)
                .map_err(VulkanError::from)?
        };

        Ok((debug_utils, debug_messenger))
    }
}

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message = CStr::from_ptr(callback_data.p_message).to_string_lossy();

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("Vulkan: {}", message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("Vulkan: {}", message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("Vulkan: {}", message),
        _ => log::debug!("Vulkan: {}", message),
    }

    vk::FALSE
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            if let (Some(debug_utils), Some(debug_messenger)) = 
                (&self.debug_utils, &self.debug_messenger) {
                debug_utils.destroy_debug_utils_messenger(*debug_messenger, None);
            }
            
            self.instance.destroy_instance(None);
        }
    }
}
