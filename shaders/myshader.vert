#version 330

uniform mat4 translation;
uniform mat4 rotation;
uniform mat4 projection;

in vec3 in_vert;
in vec3 in_color;

out vec3 v_vert;
out vec3 v_color;

void main() {
	v_vert = (translation * rotation * vec4(in_vert, 1.0)).xyz;
	v_color = in_color;
	gl_Position = translation * rotation * vec4(in_vert, 1.0);
	//gl_Position.z -= 1;
	gl_Position.z -= 10.0;

    gl_Position = projection * gl_Position;

	gl_Position.x = gl_Position.x / (gl_Position.z * 112) - 1;
	gl_Position.y = gl_Position.y / (gl_Position.z * 112) - 1;
	gl_Position.z /= 10.0;
}

