#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) flat in vec4 fragSpecular;
#ifdef SHADOWS_ENABLED
layout(location = 5) in vec4 fragLightSpacePos;
#endif

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

#ifdef SHADOWS_ENABLED
// Shadow set (set 3 in the main pipeline's own numbering), binding 0: depth-comparison sampler.
// The hardware's built-in LESS_OR_EQUAL compare (set up on the sampler/image, not here) does the
// actual hard-shadow test when sampled below -- this shader does no manual depth comparison.
layout(set = 3, binding = 0) uniform sampler2DShadow shadowMap;

// Mirrors the vertex shader's set 3 binding 1 declaration (see triangle.vert) for descriptor-set
// consistency; renamed to shadowLightViewProj here since `light` already names the directional/
// point-light LightUniforms block below. This fragment shader doesn't need to re-project
// anything itself -- fragLightSpacePos already arrives pre-transformed per-vertex.
layout(set = 3, binding = 1, row_major) uniform LightViewProj {
    mat4 lightViewProj;
} shadowLightViewProj;
#endif

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

#ifdef SHADOWS_ENABLED
// A w this small or non-positive means the light-space projection is degenerate for this
// fragment; per spec, that must not produce a black/garbage shadow, so treat it as fully lit
// rather than dividing by (near-)zero.
const float MIN_SAFE_LIGHT_SPACE_W = 1e-6;

// Fully lit (1.0) for degenerate projections and for receivers outside the light's frustum, per
// spec's "explicitly fully lit" rule. Otherwise samples the depth-comparison texture at the
// receiver's own depth; the sampler's built-in LESS_OR_EQUAL hardware compare against the stored
// occluder depth does the actual hard-shadow test and already returns 0.0 or 1.0 -- no manual
// <= comparison is done here. Raster depth bias (dynamic state, set per-frame outside this
// shader) handles shadow acne; this function applies no additional ad hoc bias.
float shadowVisibility(vec4 lightSpacePos) {
    if (lightSpacePos.w <= MIN_SAFE_LIGHT_SPACE_W) {
        return 1.0;
    }
    vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
    // Vulkan's light projection already follows Vulkan clip-space conventions -- do not flip Y a
    // second time here. Map x/y from [-1, 1] to the [0, 1] texture coordinates the sampler needs.
    vec2 uv = proj.xy * 0.5 + 0.5;
    float z = proj.z;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z < 0.0 || z > 1.0) {
        return 1.0;
    }
    return texture(shadowMap, vec3(uv, z));
}
#endif

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
#ifdef SHADOWS_ENABLED
    // Shadow visibility multiplies ONLY the directional diffuse and specular contributions;
    // ambient (below) and the point-light loop stay unshadowed, per spec.
    float visibility = shadowVisibility(fragLightSpacePos);
    vec3 shaded = visibility * (light.color * diff * albedo.rgb + highlight);
#else
    vec3 shaded = light.color * diff * albedo.rgb + highlight;
#endif

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
