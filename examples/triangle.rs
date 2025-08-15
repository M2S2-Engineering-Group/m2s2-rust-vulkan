use m2s2_rust_vulkan::{init_logging, Window, VulkanRenderer};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    
    log::info!("Creating window...");
    let mut window = Window::new("M2S2 Vulkan Triangle", 800, 600)?;
    
    log::info!("Initializing Vulkan renderer...");
    let mut renderer = VulkanRenderer::new(&window)?;
    
    let event_loop = window.take_event_loop().unwrap();

    log::info!("Starting render loop...");
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                log::info!("Window close requested");
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(physical_size),
                ..
            } => {
                log::info!("Window resized to: {:?}", physical_size);
                // TODO: Handle swapchain recreation
            }
            Event::MainEventsCleared => {
                // Render frame
                if let Err(e) = renderer.render_frame() {
                    log::error!("Render error: {}", e);
                }
            }
            _ => {}
        }
    });
}
