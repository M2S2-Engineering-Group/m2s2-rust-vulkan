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
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragWorldPos;

void main() {
    vec4 worldPos = object.model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;
    fragColor = inColor;
    fragUV = inUV;
    fragWorldPos = worldPos.xyz;
    // Normal matrix (inverse-transpose of the model's upper-left 3x3) keeps normals perpendicular
    // to the surface under non-uniform scale; correct for rotation/translation too.
    mat3 normalMatrix = transpose(inverse(mat3(object.model)));
    // Interpolating two already-unit vectors across a triangle doesn't yield a unit vector at each
    // pixel anyway, and pre-normalizing here would discard the relative-magnitude information
    // non-uniform scale encodes in the transformed normal. Normalize once, after interpolation, in
    // the fragment shader.
    fragNormal = normalMatrix * inNormal;
}
