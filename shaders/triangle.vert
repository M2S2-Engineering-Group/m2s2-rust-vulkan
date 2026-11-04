#version 450

// Per-frame MVP (spec 006). row_major: m2s2-math stores matrices row-major (data[row*cols+col]),
// but GLSL defaults to column-major — declare row_major so the bytes are read as uploaded.
layout(set = 0, binding = 0, row_major) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Vertex attributes, fed from a vertex buffer (see src/renderer/buffer.rs Vertex).
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}
