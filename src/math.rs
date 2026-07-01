pub use m2s2_math::{
    Matrix4x4f32 as Mat4, 
    Vector2f32 as Vec2, 
    Vector3f32 as Vec3, 
    Vector4f32 as Vec4,
    Transform4x4, Vector3Ops, // Import the traits
};

// Type aliases for convenience
pub type Point2f = Vec2;
pub type Point3f = Vec3;

/// Convert degrees to radians
pub fn to_radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

/// Create a perspective projection matrix (right-handed, Z in [0, 1] — Vulkan convention)
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh_zo(fov_y, aspect, near, far)
}

/// Create a look-at view matrix (right-handed — Vulkan convention)
pub fn look_at(eye: &Point3f, target: &Point3f, up: &Vec3) -> Mat4 {
    Mat4::look_at_rh(*eye, *target, *up)
}

/// Create a translation matrix
pub fn translate(translation: &Vec3) -> Mat4 {
    Mat4::translation(*translation)
}

/// Create a rotation matrix from axis and angle
pub fn rotate(axis: &Vec3, angle: f32) -> Mat4 {
    Mat4::rotation_axis_angle(*axis, angle)
}
