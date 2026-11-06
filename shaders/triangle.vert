#version 450

// m2s2-math stores matrices row-major; both blocks must read that byte layout.
layout(set = 0, binding = 0, row_major) uniform CameraUniforms {
    mat4 view;
    mat4 proj;
} camera;

layout(push_constant, row_major) uniform ModelPushConstants {
    mat4 model;
} object;

// Vertex attributes, fed from a vertex buffer (see src/renderer/buffer.rs Vertex).
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;

void main() {
    gl_Position = camera.proj * camera.view * object.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragUV = inUV;
}
