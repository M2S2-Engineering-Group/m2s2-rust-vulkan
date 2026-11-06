use m2s2_rust_vulkan::math::{
    look_at, perspective, rotate, to_radians, translate, Vec2, Vec3, Vec4,
};
use m2s2_rust_vulkan::{
    init_logging, Diagnostics, Index, Object, TextDraw, Vertex, VulkanRenderer,
};
use std::time::Instant;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::WindowBuilder;

// Leaves clearance between the unit cubes throughout their rotations.
const CUBE_CENTER_OFFSET: f32 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    log::info!("Creating window...");
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("M2S2 Vulkan Cubes")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)?;

    log::info!("Initializing Vulkan renderer...");
    let size = window.inner_size();

    // Geometry lives in the consumer, not the library (spec 006): a unit cube centered at the
    // origin. 8 corners, one color each; 36 indices (12 triangles) wound CCW-outward to match the
    // pipeline's counter-clockwise front face + back-face culling.
    let vertices = [
        Vertex {
            position: Vec3::new(-0.5, -0.5, -0.5),
            color: Vec3::new(0.0, 0.0, 0.0),
            uv: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.5, -0.5, -0.5),
            color: Vec3::new(1.0, 0.0, 0.0),
            uv: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.5, 0.5, -0.5),
            color: Vec3::new(1.0, 1.0, 0.0),
            uv: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec3::new(-0.5, 0.5, -0.5),
            color: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.0, 1.0),
        },
        Vertex {
            position: Vec3::new(-0.5, -0.5, 0.5),
            color: Vec3::new(0.0, 0.0, 1.0),
            uv: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.5, -0.5, 0.5),
            color: Vec3::new(1.0, 0.0, 1.0),
            uv: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.5, 0.5, 0.5),
            color: Vec3::new(1.0, 1.0, 1.0),
            uv: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec3::new(-0.5, 0.5, 0.5),
            color: Vec3::new(0.0, 1.0, 1.0),
            uv: Vec2::new(0.0, 1.0),
        },
    ];
    #[rustfmt::skip]
    let indices: [Index; 36] = [
        4, 5, 6,  6, 7, 4, // front (+Z)
        1, 0, 3,  3, 2, 1, // back  (-Z)
        0, 4, 7,  7, 3, 0, // left  (-X)
        5, 1, 2,  2, 6, 5, // right (+X)
        3, 7, 6,  6, 2, 3, // top   (+Y)
        0, 1, 5,  5, 4, 0, // bottom(-Y)
    ];

    let mut renderer = VulkanRenderer::new(
        window.display_handle()?.as_raw(),
        window.window_handle()?.as_raw(),
        (size.width, size.height),
        &vertices,
        &indices,
    )?;

    // The "simulation" side: a start time the cube's model matrix spins off of. This is what a game
    // engine's update step would own; the renderer only receives the resulting matrices (spec 006).
    let start = Instant::now();
    let mut last_frame = start;
    let mut diagnostics = Diagnostics::default();
    let mut show_diagnostics = true;

    log::info!("Starting render loop...");
    event_loop.run(move |event, elwt| {
        // Poll (not Wait): animate continuously, driving a frame whenever the loop is idle.
        elwt.set_control_flow(ControlFlow::Poll);

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
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } if event.state == ElementState::Pressed
                && !event.repeat
                && event.physical_key == PhysicalKey::Code(KeyCode::F3) =>
            {
                show_diagnostics = !show_diagnostics;
            }
            Event::AboutToWait => {
                let now = Instant::now();
                diagnostics.update(now.duration_since(last_frame));
                last_frame = now;
                let size = window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return; // minimized; nothing to draw
                }
                let aspect = size.width as f32 / size.height as f32;

                // Two copies of one mesh, independently transformed under a shared camera (spec 007).
                let angle = start.elapsed().as_secs_f32();
                let models = [
                    translate(&Vec3::new(-CUBE_CENTER_OFFSET, 0.0, 0.0))
                        * rotate(&Vec3::new(0.0, 1.0, 0.0), angle),
                    translate(&Vec3::new(CUBE_CENTER_OFFSET, 0.0, -0.4))
                        * rotate(&Vec3::new(1.0, 0.0, 0.0), -angle * 0.7),
                ];
                let objects = [
                    Object {
                        model: models[0],
                        texture: None,
                    },
                    Object {
                        model: models[1],
                        texture: None,
                    },
                ];
                let camera_position = Vec3::new(2.0, 2.0, 2.0);
                let view = look_at(
                    &camera_position,
                    &Vec3::new(0.0, 0.0, 0.0),
                    &Vec3::new(0.0, 1.0, 0.0),
                );
                let proj = perspective(to_radians(45.0), aspect, 0.1, 10.0);

                let diagnostic_text = diagnostics.format(&renderer.stats(), camera_position);
                let text = [TextDraw {
                    text: &diagnostic_text,
                    position: Vec2::new(12.0, 12.0),
                    scale: window.scale_factor() as f32,
                    color: Vec4::new(1.0, 1.0, 1.0, 1.0),
                }];
                let text = if show_diagnostics { &text[..] } else { &[] };
                if let Err(e) = renderer.render_objects_with_text(&objects, view, proj, text) {
                    log::error!("Render error: {e}");
                    elwt.exit();
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
