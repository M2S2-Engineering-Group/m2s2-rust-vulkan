use m2s2_rust_vulkan::init_logging;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    log::info!("Creating window...");
    let event_loop = EventLoop::new()?;
    // Kept alive for the duration of the event loop; the OS window closes when it drops.
    let _window = WindowBuilder::new()
        .with_title("M2S2 Vulkan Window")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)?;

    log::info!("Starting event loop...");
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
            }
            _ => {}
        }
    })?;

    Ok(())
}
