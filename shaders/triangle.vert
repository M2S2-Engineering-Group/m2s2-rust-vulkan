#version 450

// m2s2-math stores matrices row-major; both blocks must read that byte layout.
layout(set = 0, binding = 0, row_major) uniform CameraUniforms {
    mat4 view;
    mat4 proj;
} camera;

#ifdef SHADOWS_ENABLED
// Shadow set (set 3 in the main pipeline's own numbering): binding 0 is the depth-comparison
// sampler (fragment-only, declared in triangle.frag); binding 1 is the light's view-projection
// matrix, needed here to compute each vertex's light-space position for the fragment shader to
// sample against the shadow map.
layout(set = 3, binding = 1, row_major) uniform LightViewProj {
    mat4 lightViewProj;
} light;
#endif

// Vertex attributes, fed from a vertex buffer (see src/renderer/buffer.rs Vertex).
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;

// Per-instance attributes, fed from the per-frame instance buffer (see
// src/renderer/pipeline.rs InstanceData / src/renderer/instance_buffer.rs, plan 016). The model
// matrix arrives as four row vectors (not one mat4 attribute) because a GLSL mat4 attribute would
// consume four consecutive locations anyway, and reading it row-by-row matches InstanceData's
// GPU-side layout directly.
layout(location = 4) in vec4 inModelRow0;
layout(location = 5) in vec4 inModelRow1;
layout(location = 6) in vec4 inModelRow2;
layout(location = 7) in vec4 inModelRow3;
layout(location = 8) in vec4 inSpecular;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragWorldPos;
layout(location = 4) flat out vec4 fragSpecular;
#ifdef SHADOWS_ENABLED
layout(location = 5) out vec4 fragLightSpacePos;
#endif

void main() {
    // GLSL's mat4(c0, c1, c2, c3) constructor treats its vec4 arguments as COLUMNS. inModelRow0..3
    // are actual matrix ROWS (InstanceData extracts them via m2s2-math's row-major Index impl), so
    // building the matrix this way and then transposing it recovers the original row-major matrix
    // -- skipping the transpose would silently use the transposed matrix instead.
    mat4 model = transpose(mat4(inModelRow0, inModelRow1, inModelRow2, inModelRow3));
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;
    fragColor = inColor;
    fragUV = inUV;
    fragWorldPos = worldPos.xyz;
    // Normal matrix (inverse-transpose of the model's upper-left 3x3) keeps normals perpendicular
    // to the surface under non-uniform scale; correct for rotation/translation too.
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    // Interpolating two already-unit vectors across a triangle doesn't yield a unit vector at each
    // pixel anyway, and pre-normalizing here would discard the relative-magnitude information
    // non-uniform scale encodes in the transformed normal. Normalize once, after interpolation, in
    // the fragment shader.
    fragNormal = normalMatrix * inNormal;
    fragSpecular = inSpecular;
#ifdef SHADOWS_ENABLED
    fragLightSpacePos = light.lightViewProj * worldPos;
#endif
}
