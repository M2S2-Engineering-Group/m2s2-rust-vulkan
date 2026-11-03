use m2s2_rust_vulkan::math::Vec3;
use m2s2_rust_vulkan::{init_logging, Index, Vertex, VulkanRenderer};
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

    // Geometry lives in the consumer, not the library: a quad (two triangles) with a distinct
    // color per corner. Positions are in Vulkan clip space (Y points down); the index winding
    // matches the pipeline's clockwise front face + back-face culling.
    let vertices = [
        Vertex {
            position: Vec3::new(-0.5, -0.5, 0.0),
            color: Vec3::new(1.0, 0.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.5, -0.5, 0.0),
            color: Vec3::new(0.0, 1.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.5, 0.5, 0.0),
            color: Vec3::new(0.0, 0.0, 1.0),
        },
        Vertex {
            position: Vec3::new(-0.5, 0.5, 0.0),
            color: Vec3::new(1.0, 1.0, 0.0),
        },
    ];
    let indices: [Index; 6] = [0, 1, 2, 2, 3, 0];

    let mut renderer = VulkanRenderer::new(
        window.display_handle()?.as_raw(),
        window.window_handle()?.as_raw(),
        (size.width, size.height),
        &vertices,
        &indices,
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
