# Math Functions Needed for Vulkan Renderer

These functions need to be implemented in the m2s2-math library:

## Matrix4x4 Functions

### 1. Perspective Projection Matrix
```rust
impl Matrix4x4f32 {
    pub fn perspective(fov_y_radians: f32, aspect_ratio: f32, near: f32, far: f32) -> Self
}
```

### 2. Orthographic Projection Matrix
```rust
impl Matrix4x4f32 {
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self
}
```

### 3. Look-At View Matrix
```rust
impl Matrix4x4f32 {
    pub fn look_at(eye: Vector3f32, target: Vector3f32, up: Vector3f32) -> Self
}
```

### 4. Translation Matrix
```rust
impl Matrix4x4f32 {
    pub fn translation(translation: Vector3f32) -> Self
}
```

### 5. Rotation Matrices
```rust
impl Matrix4x4f32 {
    pub fn rotation_x(angle_radians: f32) -> Self
    pub fn rotation_y(angle_radians: f32) -> Self  
    pub fn rotation_z(angle_radians: f32) -> Self
    pub fn rotation_axis_angle(axis: Vector3f32, angle_radians: f32) -> Self
}
```

### 6. Scale Matrix
```rust
impl Matrix4x4f32 {
    pub fn scale(scale: Vector3f32) -> Self
    pub fn uniform_scale(scale: f32) -> Self
}
```

## Vector3 Functions

### 1. Vector Operations
```rust
impl Vector3f32 {
    pub fn normalize(&self) -> Self
    pub fn length(&self) -> f32
    pub fn length_squared(&self) -> f32
    pub fn dot(&self, other: &Self) -> f32
    pub fn cross(&self, other: &Self) -> Self
}
```

## Implementation Priority

1. **High Priority** (needed for basic rendering):
   - Vector3::normalize, dot, cross, length
   - Matrix4x4::perspective
   - Matrix4x4::look_at
   - Matrix4x4::translation

2. **Medium Priority** (needed for transformations):
   - Matrix4x4::rotation_* functions
   - Matrix4x4::scale

3. **Low Priority** (nice to have):
   - Matrix4x4::orthographic

## Mathematical Formulas

### Perspective Matrix (Right-handed, Vulkan NDC)
```
f = 1.0 / tan(fov_y / 2.0)
[f/aspect,  0,                    0,                     0]
[0,         f,                    0,                     0]  
[0,         0,    far/(near-far), (near*far)/(near-far)]
[0,         0,                   -1,                     0]
```

### Look-At Matrix
```
forward = normalize(target - eye)
right = normalize(cross(forward, up))
up = cross(right, forward)

[right.x,   up.x,   -forward.x,  0]
[right.y,   up.y,   -forward.y,  0]
[right.z,   up.z,   -forward.z,  0]
[-dot(right,eye), -dot(up,eye), dot(forward,eye), 1]
```
