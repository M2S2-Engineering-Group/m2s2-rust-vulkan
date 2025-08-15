use winit::{
    event_loop::EventLoop,
    window::{Window as WinitWindow, WindowBuilder},
};
use crate::error::{Result, VulkanError};

pub struct Window {
    pub window: WinitWindow,
    pub event_loop: Option<EventLoop<()>>,
}

impl Window {
    pub fn new(title: &str, width: u32, height: u32) -> Result<Self> {
        let event_loop = EventLoop::new().map_err(|e| VulkanError::WindowError(e.to_string()))?;
        
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .with_resizable(true)
            .build(&event_loop)?;

        Ok(Self {
            window,
            event_loop: Some(event_loop),
        })
    }

    pub fn take_event_loop(&mut self) -> Option<EventLoop<()>> {
        self.event_loop.take()
    }

    pub fn inner_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.window.inner_size()
    }

    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }
}
