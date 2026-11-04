use crate::error::{Result, VulkanError};
use ash::{vk, Entry, Instance};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::CStr;

#[cfg(debug_assertions)]
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

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
            .map_err(|e| VulkanError::InitializationError(format!("Failed to load Vulkan: {e}")))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"M2S2 Vulkan Renderer")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"M2S2 Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);

        let mut extension_names = vec![
            ash::khr::surface::NAME.as_ptr(),
            // Required by the device's VK_KHR_dynamic_rendering on a 1.0-API instance.
            ash::khr::get_physical_device_properties2::NAME.as_ptr(),
        ];

        // Platform-specific surface extensions
        #[cfg(target_os = "windows")]
        extension_names.push(ash::khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "macos")]
        extension_names.push(ash::ext::metal_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        extension_names.push(ash::khr::xlib_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        extension_names.push(ash::khr::wayland_surface::NAME.as_ptr());

        #[cfg_attr(not(debug_assertions), allow(unused_mut))]
        let mut layer_names = vec![];

        #[cfg(debug_assertions)]
        {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
            // Enabling an absent layer fails create_instance, so gate on availability (CI and
            // SDK-less machines have no validation layer).
            if Self::validation_layer_available(&entry) {
                layer_names.push(VALIDATION_LAYER.as_ptr());
                log::info!("Vulkan validation layer enabled");
            } else {
                log::warn!("VK_LAYER_KHRONOS_validation not available; running without validation");
            }
        }

        // In debug builds, also chain a messenger into p_next so validation messages emitted
        // *during* create_instance / destroy_instance are caught (the persistent messenger below
        // only covers the window in between).
        #[cfg(debug_assertions)]
        let mut debug_create_info = Self::debug_messenger_create_info();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);
        #[cfg(debug_assertions)]
        let create_info = create_info.push_next(&mut debug_create_info);

        let instance =
            unsafe { entry.create_instance(&create_info, None) }.map_err(VulkanError::from)?;

        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) = match Self::setup_debug_messenger(&entry, &instance) {
            Ok(messenger) => messenger,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };

        Ok(Self {
            entry,
            instance,
            #[cfg(debug_assertions)]
            debug_utils: Some(debug_utils),
            #[cfg(debug_assertions)]
            debug_messenger: Some(debug_messenger),
        })
    }

    /// Creates a `vk::SurfaceKHR` for the given window. Callers must destroy it (via the
    /// `VK_KHR_surface` instance functions) no earlier than any swapchain built from it.
    pub fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<vk::SurfaceKHR> {
        unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle,
                window_handle,
                None,
            )
        }
        .map_err(VulkanError::from)
    }

    #[cfg(debug_assertions)]
    fn validation_layer_available(entry: &Entry) -> bool {
        unsafe { entry.enumerate_instance_layer_properties() }
            .unwrap_or_default()
            .iter()
            .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER))
    }

    #[cfg(debug_assertions)]
    fn debug_messenger_create_info<'a>() -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
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
            .pfn_user_callback(Some(debug_callback))
    }

    #[cfg(debug_assertions)]
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        let debug_utils = ash::ext::debug_utils::Instance::new(entry, instance);
        let create_info = Self::debug_messenger_create_info();
        let debug_messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .map_err(VulkanError::from)?
        };

        Ok((debug_utils, debug_messenger))
    }
}

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message = CStr::from_ptr(callback_data.p_message).to_string_lossy();

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("Vulkan: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("Vulkan: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("Vulkan: {message}"),
        _ => log::debug!("Vulkan: {message}"),
    }

    vk::FALSE
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            if let (Some(debug_utils), Some(debug_messenger)) =
                (&self.debug_utils, &self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(*debug_messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}
