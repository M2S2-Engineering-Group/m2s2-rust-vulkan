use m2s2_rust_vulkan::{init_logging, VulkanRenderer};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    log::info!("Creating window...");
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("M2S2 Vulkan Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)?;

    log::info!("Initializing Vulkan renderer...");
    let size = window.inner_size();
    let mut renderer = VulkanRenderer::new(
        window.display_handle()?.as_raw(),
        window.window_handle()?.as_raw(),
        (size.width, size.height),
    )?;

    log::info!("Starting render loop...");
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Wait);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                log::info!("Window close requested");
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(physical_size),
                ..
            } => {
                log::info!("Window resized to: {physical_size:?}");
                // TODO: Handle swapchain recreation
            }
            Event::AboutToWait => {
                if let Err(e) = renderer.render_frame() {
                    log::error!("Render error: {e}");
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
