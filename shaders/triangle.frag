#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;

layout(set = 1, binding = 0) uniform sampler2D tex;

// Directional ("sun") light (see src/renderer/light.rs). std140: three vec3s at 0/16/32.
layout(set = 2, binding = 0) uniform LightUniforms {
    vec3 direction;
    vec3 color;
    vec3 ambient;
} light;

// Minimal camera block: only the world-space eye position is needed here (spec 013); the vertex
// stage's view/proj members at offsets 0/64 are not redeclared.
layout(set = 0, binding = 0) uniform CameraUniforms {
    layout(offset = 128) vec3 position;
} camera;

// Material push-constant subset: the vertex stage's `model` member at offset 0 is not redeclared.
layout(push_constant) uniform MaterialPushConstants {
    layout(offset = 64) vec3 specularColor;
    layout(offset = 76) float shininess;
} material;

layout(location = 0) out vec4 outColor;

const float MIN_SAFE_LENGTH_SQUARED = 1e-12;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-light.direction);            // surface -> light
    float diff = max(dot(N, L), 0.0);                // Lambert
    vec3 shade = light.ambient + light.color * diff; // ambient floor + diffuse
    vec4 albedo = texture(tex, fragUV) * vec4(fragColor, 1.0);

    // Blinn-Phong specular highlight: only when the surface faces both the light and the camera,
    // and only when the view/halfway vectors are long enough to normalize safely.
    vec3 highlight = vec3(0.0);
    vec3 toCamera = camera.position - fragWorldPos;
    float viewLenSq = dot(toCamera, toCamera);
    if (diff > 0.0 && viewLenSq > MIN_SAFE_LENGTH_SQUARED) {
        vec3 V = toCamera * inversesqrt(viewLenSq);
        if (dot(N, V) > 0.0) {
            vec3 halfway = L + V;
            float halfLenSq = dot(halfway, halfway);
            if (halfLenSq > MIN_SAFE_LENGTH_SQUARED) {
                vec3 H = halfway * inversesqrt(halfLenSq);
                highlight = light.color * material.specularColor * pow(max(dot(N, H), 0.0), material.shininess);
            }
        }
    }
    outColor = vec4(shade, 1.0) * albedo + vec4(highlight, 0.0);
}
