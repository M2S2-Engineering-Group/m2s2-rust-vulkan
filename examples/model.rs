use m2s2_rust_vulkan::math::{look_at, perspective, to_radians, Vec2, Vec3, Vec4};
use m2s2_rust_vulkan::{
    init_logging, model, Camera, Diagnostics, DirectionalLight, Object, SpecularMaterial, TextDraw,
    TextureId, VulkanRenderer,
};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

// Free-fly camera tuning (same scheme as examples/cube.rs).
const MOVE_SPEED: f32 = 20.0; // world units per second — the mech spans ~30 units
const LOOK_SPEED: f32 = 1.5; // radians per second
const MAX_CAMERA_DELTA: Duration = Duration::from_millis(100);
const MAX_PITCH: f32 = 1.553;

// The mech's own world-space bounding box (computed once from the converted asset) places its
// center near (1.3, 11.5, 2.2) and spans roughly 32x22x18 units — the initial camera sits back
// along +Z from that center so the whole model is in view on launch.
const LOOK_AT_TARGET: [f32; 3] = [1.3, 11.5, 2.2];
const INITIAL_CAMERA_OFFSET: [f32; 3] = [0.0, 5.0, 55.0];

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    log::info!("Creating window...");
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("M2S2 Vulkan Model Loading (Mecha-Ziu)")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            .build(&event_loop)?,
    );

    log::info!("Initializing Vulkan renderer...");
    let size = window.inner_size();
    let mut renderer = VulkanRenderer::with_window(Arc::clone(&window), (size.width, size.height))?;

    // Spec 020: load_gltf returns plain geometry/transform/texture-reference data; this example
    // owns turning that into renderer resources exactly like it would for hand-written geometry.
    let asset_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/mecha_ziu/MechaZiu.glb");
    let loaded = model::load_gltf(&asset_path)?;

    let mut texture_ids: Vec<TextureId> = Vec::with_capacity(loaded.textures.len());
    for texture in &loaded.textures {
        let decoder = png::Decoder::new(std::io::Cursor::new(&texture.bytes));
        let mut reader = decoder.read_info()?;
        let mut pixels = vec![0u8; reader.output_buffer_size()];
        reader.next_frame(&mut pixels)?;
        let info = reader.info();
        texture_ids.push(renderer.create_texture(&pixels, info.width, info.height)?);
    }

    let mut objects = Vec::with_capacity(loaded.meshes.len());
    for loaded_mesh in &loaded.meshes {
        let mesh = renderer.create_mesh(&loaded_mesh.vertices, &loaded_mesh.indices)?;
        objects.push(Object {
            mesh,
            model: loaded_mesh.world_transform,
            texture: loaded_mesh.texture.map(|i| texture_ids[i]),
            specular: SpecularMaterial::default(),
        });
    }
    log::info!(
        "Loaded {} meshes, {} textures from {}",
        objects.len(),
        texture_ids.len(),
        asset_path.display()
    );

    renderer.set_directional_light(DirectionalLight {
        direction: Vec3::new(-0.4, -1.0, -0.3),
        color: Vec3::new(0.9, 0.88, 0.85),
        ambient: Vec3::new(0.15, 0.15, 0.17),
    })?;

    let mut last_frame = Instant::now();
    let mut diagnostics = Diagnostics::default();
    let mut show_diagnostics = true;

    let initial_pos = [
        LOOK_AT_TARGET[0] + INITIAL_CAMERA_OFFSET[0],
        LOOK_AT_TARGET[1] + INITIAL_CAMERA_OFFSET[1],
        LOOK_AT_TARGET[2] + INITIAL_CAMERA_OFFSET[2],
    ];
    let (dx, dy, dz) = (
        LOOK_AT_TARGET[0] - initial_pos[0],
        LOOK_AT_TARGET[1] - initial_pos[1],
        LOOK_AT_TARGET[2] - initial_pos[2],
    );
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
                    return; // minimized
                }
                let aspect = size.width as f32 / size.height as f32;

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
                let proj = perspective(to_radians(45.0), aspect, 0.1, 200.0);

                let diagnostic_text = diagnostics.format(&renderer.stats(), camera_position);
                let text = [TextDraw {
                    text: &diagnostic_text,
                    position: Vec2::new(12.0, 12.0),
                    scale: window.scale_factor() as f32,
                    color: Vec4::new(1.0, 1.0, 1.0, 1.0),
                }];
                let text = if show_diagnostics { &text[..] } else { &[] };
                let camera = Camera {
                    view,
                    proj,
                    position: camera_position,
                };
                if let Err(e) = renderer.render_objects_with_text(&objects, camera, text) {
                    log::error!("Render error: {e}");
                    elwt.exit();
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
