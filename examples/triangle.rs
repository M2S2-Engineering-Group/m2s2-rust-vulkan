// The minimal example: a single static triangle. Start here to see the smallest end-to-end use of
// the library. For a 3D scene (per-frame MVP uniforms, a moving model matrix, depth testing) see
// examples/cube.rs.

use m2s2_rust_vulkan::math::{look_at, perspective, to_radians, Mat4, Vec2, Vec3};
use m2s2_rust_vulkan::{init_logging, Camera, Index, Vertex, VulkanRenderer};
use std::sync::Arc;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    log::info!("Creating window...");
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("M2S2 Vulkan Triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)?,
    );

    log::info!("Initializing Vulkan renderer...");
    let size = window.inner_size();

    // Geometry lives in the consumer, not the library. One triangle in the z=0 plane, a distinct
    // color per corner. Wound counter-clockwise as seen from +Z (where the camera sits) to match the
    // pipeline's counter-clockwise front face + back-face culling.
    let vertices = [
        Vertex {
            position: Vec3::new(-0.5, -0.5, 0.0),
            color: Vec3::new(1.0, 0.0, 0.0),
            uv: Vec2::new(0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
        },
        Vertex {
            position: Vec3::new(0.5, -0.5, 0.0),
            color: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
        },
        Vertex {
            position: Vec3::new(0.0, 0.5, 0.0),
            color: Vec3::new(0.0, 0.0, 1.0),
            uv: Vec2::new(0.5, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
        },
    ];
    let indices: [Index; 3] = [0, 1, 2];

    let mut renderer = VulkanRenderer::with_window(
        Arc::clone(&window),
        (size.width, size.height),
        &vertices,
        &indices,
    )?;

    log::info!("Starting render loop...");
    event_loop.run(move |event, elwt| {
        // Wait: the triangle is static, so only redraw in response to events (no busy loop).
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
                renderer.resize((physical_size.width, physical_size.height));
            }
            Event::AboutToWait => {
                let size = window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return; // minimized; nothing to draw
                }
                let aspect = size.width as f32 / size.height as f32;

                // A fixed camera looking at the origin from +Z; the triangle doesn't move, so its
                // model matrix is the identity.
                let model = Mat4::IDENTITY;
                let camera_position = Vec3::new(0.0, 0.0, 2.0);
                let view = look_at(
                    &camera_position,
                    &Vec3::new(0.0, 0.0, 0.0),
                    &Vec3::new(0.0, 1.0, 0.0),
                );
                let proj = perspective(to_radians(45.0), aspect, 0.1, 10.0);

                if let Err(e) = renderer.render_frame(
                    model,
                    Camera {
                        view,
                        proj,
                        position: camera_position,
                    },
                ) {
                    log::error!("Render error: {e}");
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
