#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;

layout(set = 1, binding = 0) uniform sampler2D tex;

// Directional ("sun") light (see src/renderer/light.rs). std140: three vec3s at 0/16/32.
layout(set = 2, binding = 0) uniform LightUniforms {
    vec3 direction;
    vec3 color;
    vec3 ambient;
} light;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-light.direction);            // surface -> light
    float diff = max(dot(N, L), 0.0);                // Lambert
    vec3 shade = light.ambient + light.color * diff; // ambient floor + diffuse
    vec4 albedo = texture(tex, fragUV) * vec4(fragColor, 1.0);
    outColor = vec4(shade, 1.0) * albedo;
}
