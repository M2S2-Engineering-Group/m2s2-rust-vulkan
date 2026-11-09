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
    /// Backing counter for [`VulkanInstance::validation_error_count`]. Always `None` for the
    /// production `new()` constructor; `Some(...)` only for the test-only `new_headless()`, whose
    /// debug-callback `user_data` points at this same `Arc`'s counter. `#[cfg(test)]` in addition
    /// to `debug_assertions` because its only reader, `validation_error_count`, is itself
    /// `#[cfg(test)]` — without that, a non-test debug build would see the field as dead code.
    #[cfg(all(test, debug_assertions))]
    validation_errors: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
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
            // Vulkan 1.3: makes dynamic rendering core (spec 009). A 1.0-only loader rejects this
            // with ERROR_INCOMPATIBLE_DRIVER, which is the intended hard requirement.
            .api_version(vk::API_VERSION_1_3);

        let mut extension_names = vec![ash::khr::surface::NAME.as_ptr()];

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
        let mut debug_create_info = Self::debug_messenger_create_info(std::ptr::null_mut());

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);
        #[cfg(debug_assertions)]
        let create_info = create_info.push_next(&mut debug_create_info);

        let instance =
            unsafe { entry.create_instance(&create_info, None) }.map_err(VulkanError::from)?;

        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) =
            match Self::setup_debug_messenger(&entry, &instance, std::ptr::null_mut()) {
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
            #[cfg(all(test, debug_assertions))]
            validation_errors: None,
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
    fn debug_messenger_create_info<'a>(
        user_data: *mut std::os::raw::c_void,
    ) -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
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
            .user_data(user_data)
    }

    #[cfg(debug_assertions)]
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
        user_data: *mut std::os::raw::c_void,
    ) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        let debug_utils = ash::ext::debug_utils::Instance::new(entry, instance);
        let create_info = Self::debug_messenger_create_info(user_data);
        let debug_messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .map_err(VulkanError::from)?
        };

        Ok((debug_utils, debug_messenger))
    }

    /// Builds an instance with no surface extensions at all — for headless (no window, no
    /// swapchain) golden-image tests. In debug builds, the debug messenger's `user_data` points
    /// at a fresh `Arc<AtomicU32>` stored in `validation_errors`, so [`Self::validation_error_count`]
    /// can assert "zero validation errors" without scraping logs.
    #[cfg(test)]
    pub(crate) fn new_headless() -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| VulkanError::InitializationError(format!("Failed to load Vulkan: {e}")))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"M2S2 Vulkan Renderer (headless)")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"M2S2 Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        #[cfg_attr(not(debug_assertions), allow(unused_mut))]
        let mut extension_names = vec![];

        #[cfg_attr(not(debug_assertions), allow(unused_mut))]
        let mut layer_names = vec![];

        #[cfg(debug_assertions)]
        {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
            if Self::validation_layer_available(&entry) {
                layer_names.push(VALIDATION_LAYER.as_ptr());
                log::info!("Vulkan validation layer enabled");
            } else {
                log::warn!("VK_LAYER_KHRONOS_validation not available; running without validation");
            }
        }

        #[cfg(debug_assertions)]
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        #[cfg(debug_assertions)]
        let user_data = std::sync::Arc::as_ptr(&counter) as *mut std::os::raw::c_void;

        #[cfg(debug_assertions)]
        let mut debug_create_info = Self::debug_messenger_create_info(user_data);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);
        #[cfg(debug_assertions)]
        let create_info = create_info.push_next(&mut debug_create_info);

        let instance =
            unsafe { entry.create_instance(&create_info, None) }.map_err(VulkanError::from)?;

        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) =
            match Self::setup_debug_messenger(&entry, &instance, user_data) {
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
            #[cfg(debug_assertions)]
            validation_errors: Some(counter),
        })
    }

    /// Number of ERROR-severity validation messages observed since construction. Always `0` for
    /// an instance built via [`Self::new`] (which never installs a counter); meaningful only for
    /// [`Self::new_headless`].
    #[cfg(test)]
    #[cfg(debug_assertions)]
    pub(crate) fn validation_error_count(&self) -> u32 {
        self.validation_errors
            .as_ref()
            .map(|counter| counter.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0)
    }

    #[cfg(test)]
    #[cfg(not(debug_assertions))]
    pub(crate) fn validation_error_count(&self) -> u32 {
        0
    }
}

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message = CStr::from_ptr(callback_data.p_message).to_string_lossy();

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("Vulkan: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("Vulkan: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("Vulkan: {message}"),
        _ => log::debug!("Vulkan: {message}"),
    }

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR && !user_data.is_null() {
        let counter = &*(user_data as *const std::sync::atomic::AtomicU32);
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(debug_assertions)]
    #[test]
    fn debug_callback_increments_counter_only_on_error_severity() {
        use std::ffi::CString;
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = AtomicU32::new(0);
        let message = CString::new("test message").unwrap();
        let data = vk::DebugUtilsMessengerCallbackDataEXT::default().message(message.as_c_str());
        let user_data = &counter as *const AtomicU32 as *mut std::os::raw::c_void;

        unsafe {
            debug_callback(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                &data,
                user_data,
            );
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        unsafe {
            debug_callback(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                &data,
                user_data,
            );
        }
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_callback_tolerates_null_user_data() {
        let message = std::ffi::CString::new("test message").unwrap();
        let data = vk::DebugUtilsMessengerCallbackDataEXT::default().message(message.as_c_str());
        unsafe {
            debug_callback(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                &data,
                std::ptr::null_mut(),
            );
        } // must not panic/segfault
    }
}
