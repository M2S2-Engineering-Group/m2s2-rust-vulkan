use m2s2_rust_vulkan::math::{
    look_at, perspective, rotate, to_radians, translate, Vec2, Vec3, Vec4,
};
use m2s2_rust_vulkan::{
    init_logging, Diagnostics, DirectionalLight, Index, Object, TextDraw, Vertex, VulkanRenderer,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::WindowBuilder;

// Leaves clearance between the unit cubes throughout their rotations.
const CUBE_CENTER_OFFSET: f32 = 1.0;
// Free-fly camera tuning.
const MOVE_SPEED: f32 = 3.0; // world units per second (WASD)
const LOOK_SPEED: f32 = 1.5; // radians per second (arrow keys)
const MAX_CAMERA_DELTA: Duration = Duration::from_millis(100);
const MAX_PITCH: f32 = 1.553; // ~89 degrees, keeps the camera from flipping over the pole

struct CameraInput {
    held: HashSet<KeyCode>,
    focused: bool,
}

impl CameraInput {
    fn set_focus(&mut self, focused: bool) {
        self.focused = focused;
        if !focused {
            self.held.clear();
        }
    }

    fn set_key(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed if self.focused => {
                self.held.insert(key);
            }
            ElementState::Released => {
                self.held.remove(&key);
            }
            _ => {}
        }
    }

    fn delta(&self, elapsed: Duration) -> f32 {
        if self.focused {
            elapsed.min(MAX_CAMERA_DELTA).as_secs_f32()
        } else {
            0.0
        }
    }
}

/// Builds the four vertices of one cube face, each carrying the face's outward `normal` so lighting
/// shades per face. `corners` are the face's 4 positions in CCW order as seen from outside (matching
/// the pipeline's CCW-front + back-face culling), paired with a per-corner color; UVs map the unit
/// square. Flat per-face normals are why the cube needs 24 vertices, not 8 shared corners.
fn face(normal: [f32; 3], corners: [([f32; 3], [f32; 3]); 4]) -> [Vertex; 4] {
    const UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    std::array::from_fn(|i| {
        let (p, c) = corners[i];
        Vertex {
            position: Vec3::new(p[0], p[1], p[2]),
            color: Vec3::new(c[0], c[1], c[2]),
            uv: Vec2::new(UVS[i][0], UVS[i][1]),
            normal: Vec3::new(normal[0], normal[1], normal[2]),
        }
    })
}

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

    // Geometry lives in the consumer, not the library (spec 006): a unit cube centered at the origin,
    // built face-by-face as 24 vertices (4 per face) so each face carries its own outward normal for
    // crisp flat lighting. Per-corner colors reuse the 8-corner palette. 36 indices (6 per face).
    const BLACK: [f32; 3] = [0.0, 0.0, 0.0];
    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
    const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
    const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
    const MAGENTA: [f32; 3] = [1.0, 0.0, 1.0];
    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
    const CYAN: [f32; 3] = [0.0, 1.0, 1.0];
    let mut vertices = Vec::with_capacity(24);
    vertices.extend(face(
        [0.0, 0.0, 1.0], // front (+Z)
        [
            ([-0.5, -0.5, 0.5], BLUE),
            ([0.5, -0.5, 0.5], MAGENTA),
            ([0.5, 0.5, 0.5], WHITE),
            ([-0.5, 0.5, 0.5], CYAN),
        ],
    ));
    vertices.extend(face(
        [0.0, 0.0, -1.0], // back (-Z)
        [
            ([0.5, -0.5, -0.5], RED),
            ([-0.5, -0.5, -0.5], BLACK),
            ([-0.5, 0.5, -0.5], GREEN),
            ([0.5, 0.5, -0.5], YELLOW),
        ],
    ));
    vertices.extend(face(
        [-1.0, 0.0, 0.0], // left (-X)
        [
            ([-0.5, -0.5, -0.5], BLACK),
            ([-0.5, -0.5, 0.5], BLUE),
            ([-0.5, 0.5, 0.5], CYAN),
            ([-0.5, 0.5, -0.5], GREEN),
        ],
    ));
    vertices.extend(face(
        [1.0, 0.0, 0.0], // right (+X)
        [
            ([0.5, -0.5, 0.5], MAGENTA),
            ([0.5, -0.5, -0.5], RED),
            ([0.5, 0.5, -0.5], YELLOW),
            ([0.5, 0.5, 0.5], WHITE),
        ],
    ));
    vertices.extend(face(
        [0.0, 1.0, 0.0], // top (+Y)
        [
            ([-0.5, 0.5, 0.5], CYAN),
            ([0.5, 0.5, 0.5], WHITE),
            ([0.5, 0.5, -0.5], YELLOW),
            ([-0.5, 0.5, -0.5], GREEN),
        ],
    ));
    vertices.extend(face(
        [0.0, -1.0, 0.0], // bottom (-Y)
        [
            ([-0.5, -0.5, -0.5], BLACK),
            ([0.5, -0.5, -0.5], RED),
            ([0.5, -0.5, 0.5], MAGENTA),
            ([-0.5, -0.5, 0.5], BLUE),
        ],
    ));
    // Each face's 4 vertices (base b = face*4) triangulate as (b,b+1,b+2) + (b+2,b+3,b), CCW-outward.
    let indices: Vec<Index> = (0..6u16)
        .flat_map(|f| {
            let b = f * 4;
            [b, b + 1, b + 2, b + 2, b + 3, b]
        })
        .collect();

    let mut renderer = VulkanRenderer::new(
        window.display_handle()?.as_raw(),
        window.window_handle()?.as_raw(),
        (size.width, size.height),
        &vertices,
        &indices,
    )?;

    // Two 2x2 RGBA8 textures: a red/white checker and a blue/green checker.
    fn checker(a: [u8; 4], b: [u8; 4]) -> [u8; 16] {
        let mut px = [0u8; 16];
        for (i, color) in [a, b, b, a].into_iter().enumerate() {
            px[i * 4..i * 4 + 4].copy_from_slice(&color);
        }
        px
    }
    let tex_a = renderer
        .create_texture(&checker([255, 40, 40, 255], [255, 255, 255, 255]), 2, 2)
        .expect("create texture A");
    let tex_b = renderer
        .create_texture(&checker([40, 80, 255, 255], [40, 255, 120, 255]), 2, 2)
        .expect("create texture B");

    // A stationary sun. Faces toward it are bright; faces away fall to the ambient floor. Moving the
    // camera (WASD + arrows) reveals the lit/unlit faces from every angle.
    renderer.set_directional_light(DirectionalLight {
        direction: Vec3::new(-0.4, -1.0, -0.6),
        color: Vec3::new(1.0, 0.97, 0.9),
        ambient: Vec3::new(0.12, 0.12, 0.15),
    })?;

    // The "simulation" side: a start time the cube's model matrix spins off of. This is what a game
    // engine's update step would own; the renderer only receives the resulting matrices (spec 006).
    let start = Instant::now();
    let mut last_frame = start;
    let mut diagnostics = Diagnostics::default();
    let mut show_diagnostics = true;

    // Free-fly camera state. Initial pose sits back from the scene looking at the origin; the initial
    // yaw/pitch are derived from that look direction so "reset" (R) returns here exactly.
    let initial_pos = [2.0_f32, 2.0, 5.0];
    let (dx, dy, dz) = (-initial_pos[0], -initial_pos[1], -initial_pos[2]);
    let dlen = (dx * dx + dy * dy + dz * dz).sqrt();
    let initial_yaw = (dz / dlen).atan2(dx / dlen);
    let initial_pitch = (dy / dlen).asin();
    let (mut cam_x, mut cam_y, mut cam_z) = (initial_pos[0], initial_pos[1], initial_pos[2]);
    let mut yaw = initial_yaw;
    let mut pitch = initial_pitch;
    let mut input = CameraInput {
        held: HashSet::new(),
        focused: true,
    };

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
                event: WindowEvent::Focused(focused),
                ..
            } => {
                input.set_focus(focused);
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    input.set_key(code, event.state);
                    if event.state == ElementState::Pressed && input.focused && !event.repeat {
                        match code {
                            KeyCode::F3 => show_diagnostics = !show_diagnostics,
                            KeyCode::KeyR => {
                                cam_x = initial_pos[0];
                                cam_y = initial_pos[1];
                                cam_z = initial_pos[2];
                                yaw = initial_yaw;
                                pitch = initial_pitch;
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::AboutToWait => {
                let now = Instant::now();
                let frame_dt = now.duration_since(last_frame);
                diagnostics.update(frame_dt);
                last_frame = now;
                let dt = input.delta(frame_dt);
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
                        texture: Some(tex_a),
                    },
                    Object {
                        model: models[1],
                        texture: Some(tex_b),
                    },
                ];
                // Free-fly camera: arrow keys rotate (yaw/pitch), WASD move relative to the facing.
                if input.held.contains(&KeyCode::ArrowLeft) {
                    yaw -= LOOK_SPEED * dt;
                }
                if input.held.contains(&KeyCode::ArrowRight) {
                    yaw += LOOK_SPEED * dt;
                }
                if input.held.contains(&KeyCode::ArrowUp) {
                    pitch += LOOK_SPEED * dt;
                }
                if input.held.contains(&KeyCode::ArrowDown) {
                    pitch -= LOOK_SPEED * dt;
                }
                pitch = pitch.clamp(-MAX_PITCH, MAX_PITCH);

                let (sy, cy) = yaw.sin_cos();
                let (sp, cp) = pitch.sin_cos();
                let forward = [cp * cy, sp, cp * sy];
                // Right = normalize(forward x worldUp), worldUp = (0,1,0) -> (-fz, 0, fx).
                let rlen = (forward[2] * forward[2] + forward[0] * forward[0]).sqrt();
                let right = if rlen > 1e-6 {
                    [-forward[2] / rlen, 0.0, forward[0] / rlen]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let mut mv = [0.0_f32; 3];
                for (key, dir, sign) in [
                    (KeyCode::KeyW, forward, 1.0),
                    (KeyCode::KeyS, forward, -1.0),
                    (KeyCode::KeyD, right, 1.0),
                    (KeyCode::KeyA, right, -1.0),
                ] {
                    if input.held.contains(&key) {
                        mv[0] += dir[0] * sign;
                        mv[1] += dir[1] * sign;
                        mv[2] += dir[2] * sign;
                    }
                }
                let mlen = (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]).sqrt();
                if mlen > 1e-6 {
                    let step = MOVE_SPEED * dt / mlen;
                    cam_x += mv[0] * step;
                    cam_y += mv[1] * step;
                    cam_z += mv[2] * step;
                }

                let camera_position = Vec3::new(cam_x, cam_y, cam_z);
                let target = Vec3::new(cam_x + forward[0], cam_y + forward[1], cam_z + forward[2]);
                let view = look_at(&camera_position, &target, &Vec3::new(0.0, 1.0, 0.0));
                let proj = perspective(to_radians(45.0), aspect, 0.1, 100.0);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_loss_clears_movement_and_look_without_sticky_keys() {
        let mut input = CameraInput {
            held: HashSet::new(),
            focused: true,
        };
        input.set_key(KeyCode::KeyW, ElementState::Pressed);
        input.set_key(KeyCode::ArrowLeft, ElementState::Pressed);
        assert_eq!(input.held.len(), 2);
        input.set_focus(true);
        assert_eq!(input.held.len(), 2);
        input.set_focus(false);
        assert!(input.held.is_empty());
        assert_eq!(input.delta(Duration::from_secs(10)), 0.0);
        input.set_key(KeyCode::KeyW, ElementState::Pressed);
        assert!(input.held.is_empty());
        input.set_key(KeyCode::ArrowLeft, ElementState::Released);
        input.set_focus(true);
        assert!(input.held.is_empty());
        input.set_key(KeyCode::KeyD, ElementState::Pressed);
        assert!(input.held.contains(&KeyCode::KeyD));
        input.set_key(KeyCode::KeyD, ElementState::Released);
        assert!(input.held.is_empty());
    }

    #[test]
    fn camera_delta_preserves_normal_frames_and_bounds_stalls() {
        let input = CameraInput {
            held: HashSet::new(),
            focused: true,
        };
        for duration in [Duration::ZERO, Duration::from_millis(16), MAX_CAMERA_DELTA] {
            assert_eq!(input.delta(duration), duration.as_secs_f32());
        }
        assert_eq!(
            input.delta(Duration::from_secs(10)),
            MAX_CAMERA_DELTA.as_secs_f32()
        );
        assert!(MOVE_SPEED * input.delta(Duration::from_secs(10)) <= 0.301);
        assert!(LOOK_SPEED * input.delta(Duration::from_secs(10)) <= 0.151);
    }
}
