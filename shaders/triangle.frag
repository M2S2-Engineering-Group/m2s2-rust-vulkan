#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) flat in vec4 fragSpecular;

layout(set = 1, binding = 0) uniform sampler2D tex;

// Must match src/renderer/light.rs's MAX_POINT_LIGHTS (the `points` array length baked into
// SceneLightUniforms's std140 layout).
#define MAX_POINT_LIGHTS 8

// One point light's std140 layout (see src/renderer/light.rs PointLightUniforms): position/range at
// offsets 0/12, color at offset 16, with trailing padding to a 32-byte stride.
struct PointLight {
    vec3 position;
    float range;
    vec3 color;
};

// Directional ("sun") light plus up to MAX_POINT_LIGHTS point lights (see src/renderer/light.rs
// SceneLightUniforms). std140: direction/color/ambient at 0/16/32, pointCount at 48, points at 64.
layout(set = 2, binding = 0) uniform LightUniforms {
    vec3 direction;
    vec3 color;
    vec3 ambient;
    // std140's automatic packing would place pointCount at offset 44 (the 4 leftover bytes after
    // ambient's 12-byte vec3, since a bare uint only needs 4-byte alignment) and points at 48 —
    // NOT the Rust-side SceneLightUniforms offsets of 48/64 (src/renderer/light.rs forces those via
    // explicit padding fields). Explicit offsets here keep the two layouts in sync.
    layout(offset = 48) uint pointCount;
    layout(offset = 64) PointLight points[MAX_POINT_LIGHTS];
} light;

// Minimal camera block: only the world-space eye position is needed here (spec 013); the vertex
// stage's view/proj members at offsets 0/64 are not redeclared.
layout(set = 0, binding = 0) uniform CameraUniforms {
    layout(offset = 128) vec3 position;
} camera;

layout(location = 0) out vec4 outColor;

const float MIN_SAFE_LENGTH_SQUARED = 1e-12;

// Blinn-Phong specular highlight for one light: only when the surface faces both the light and the
// camera, and only when the halfway vector is long enough to normalize safely. N, L, V must already
// be normalized (V's own degeneracy guard is the caller's responsibility).
vec3 blinnPhongSpecular(vec3 N, vec3 L, vec3 V, vec3 specularColor, float shininess) {
    if (dot(N, L) <= 0.0 || dot(N, V) <= 0.0) {
        return vec3(0.0);
    }
    vec3 halfway = L + V;
    float halfLenSq = dot(halfway, halfway);
    if (halfLenSq <= MIN_SAFE_LENGTH_SQUARED) {
        return vec3(0.0);
    }
    vec3 H = halfway * inversesqrt(halfLenSq);
    return specularColor * pow(max(dot(N, H), 0.0), shininess);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec4 albedo = texture(tex, fragUV) * vec4(fragColor, 1.0);

    // Camera-to-surface view vector, normalized once up front; zero when degenerate so every light's
    // specular term naturally contributes nothing without each light re-deriving this guard.
    vec3 toCamera = camera.position - fragWorldPos;
    float viewLenSq = dot(toCamera, toCamera);
    vec3 V = viewLenSq > MIN_SAFE_LENGTH_SQUARED ? toCamera * inversesqrt(viewLenSq) : vec3(0.0);

    // Directional ("sun") light: Lambert diffuse + Blinn-Phong specular.
    vec3 L = normalize(-light.direction); // surface -> light
    float diff = max(dot(N, L), 0.0);
    vec3 highlight = light.color * blinnPhongSpecular(N, L, V, fragSpecular.rgb, fragSpecular.a);
    vec3 shaded = light.color * diff * albedo.rgb + highlight;

    // Point lights: inverse-square-ish falloff (t = 1 - d/range; attenuation = t^2) clamped to zero
    // at/beyond range, added on top of the directional contribution.
    for (uint i = 0u; i < light.pointCount; i++) {
        vec3 toLight = light.points[i].position - fragWorldPos;
        float distSq = dot(toLight, toLight);
        if (distSq <= MIN_SAFE_LENGTH_SQUARED) {
            continue;
        }
        float d = sqrt(distSq);
        float range = light.points[i].range;
        if (d < range) {
            vec3 pointL = toLight / d;
            float t = 1.0 - d / range;
            float attenuation = t * t;
            vec3 pointHighlight = blinnPhongSpecular(N, pointL, V, fragSpecular.rgb, fragSpecular.a);
            shaded += attenuation * light.points[i].color *
                (max(dot(N, pointL), 0.0) * albedo.rgb + pointHighlight);
        }
    }

    outColor = vec4(light.ambient * albedo.rgb + shaded, albedo.a);
}
