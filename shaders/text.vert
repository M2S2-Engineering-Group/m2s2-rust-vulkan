#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;
layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragUV = uv;
    fragColor = color;
}
