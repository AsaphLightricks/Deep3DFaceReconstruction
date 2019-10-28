#version 330

in vec3 v_vert;
in vec3 v_color;
in vec2 v_text;

out vec4 f_color;

void main() {
    f_color = vec4(v_color, 1.0);
}