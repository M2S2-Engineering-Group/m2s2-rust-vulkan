#version 450

// Depth-only shadow-pass vertex shader (spec 017). No fragment shader accompanies this one --
// rasterization writes depth automatically against the shadow map's depth attachment, so there's
// nothing for a fragment stage to compute. Reuses the main pipeline's existing vertex/instance
// bindings (see shaders/triangle.vert, src/renderer/buffer.rs Vertex, and
// src/renderer/pipeline.rs InstanceData) but declares only the inputs this shader actually reads:
// position from binding 0 and the four per-instance model-row vectors from binding 1. GLSL/Vulkan
// allow a vertex shader to declare a subset of the attributes the bound vertex-input state
// describes, so inColor/inUV/inNormal (binding 0) and specular (binding 1, location 8) are simply
// omitted.
layout(location = 0) in vec3 inPosition;

layout(location = 4) in vec4 inModelRow0;
layout(location = 5) in vec4 inModelRow1;
layout(location = 6) in vec4 inModelRow2;
layout(location = 7) in vec4 inModelRow3;

// The shadow pass's pipeline layout has exactly one descriptor set (the shadow set) at index 0 --
// even though the SAME descriptor set object is bound as "set 3" in the main pipeline's own
// numbering. Binding 0 is the depth-comparison sampler this shader never reads (only the main
// pass's fragment shader samples it); binding 1 is the light's view-projection matrix.
layout(set = 0, binding = 1, row_major) uniform LightViewProj {
    mat4 lightViewProj;
} light;

void main() {
    // GLSL's mat4(c0, c1, c2, c3) constructor treats its vec4 arguments as COLUMNS. inModelRow0..3
    // are actual matrix ROWS (InstanceData extracts them via m2s2-math's row-major Index impl), so
    // building the matrix this way and then transposing it recovers the original row-major matrix
    // -- skipping the transpose would silently use the transposed matrix instead. Same
    // reconstruction as triangle.vert.
    mat4 model = transpose(mat4(inModelRow0, inModelRow1, inModelRow2, inModelRow3));
    gl_Position = light.lightViewProj * model * vec4(inPosition, 1.0);
}
