
import moderngl
import numpy as np
from pyrr import Matrix44


def get_vertex_data(world_coords, colors, triangles):
    vertices = np.hstack([world_coords, colors])
    vert_list = []
    for v1, v2, v3 in triangles:
        vert_list.append(vertices[v1])
        vert_list.append(vertices[v2])
        vert_list.append(vertices[v3])
    vert_list = np.array(vert_list)
    return vert_list.astype('f4').tobytes()


class Renderer(object):

    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.depth_func = '<'

        # Shaders
        vertex_shader_source = open('shaders/myshader.vert').read()
        fragment_shader_source = open('shaders/myshader.frag').read()
        self.prog = self.ctx.program(vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source)

    def render(self, world_coords, colors, triangles, projection, rotation, translation, out_shape=(224, 224)):

        # Matrices and Uniforms
        # fov_rad = 2 * np.arctan(120 / focal_length)  # (film_size / 2) / focal_length
        # fov_deg = fov_rad * 180 / np.pi

        # projection_mat = Matrix44.perspective_projection(fov_deg, 1.0, 1000, 1200)

        translation_mat = Matrix44.from_translation(translation)
        projection_mat = Matrix44.from_matrix33(projection.T)
        rotation_mat = Matrix44.from_matrix33(rotation)

        translation = self.prog['translation']
        projection = self.prog['projection']
        rotation = self.prog['rotation']

        translation.write(translation_mat.astype('f4').tobytes())
        projection.write(projection_mat.astype('f4').tobytes())
        rotation.write(rotation_mat.astype('f4').tobytes())

        # vertex array and buffer (binding the mesh)
        vbo = self.ctx.buffer(get_vertex_data(world_coords, colors, triangles))
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_color')

        # frame buffer setup
        rbo = self.ctx.renderbuffer(out_shape, components=4, dtype='f4')
        dbo = self.ctx.depth_renderbuffer(out_shape)
        fbo = self.ctx.framebuffer([rbo], dbo)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        # render
        vao.render()

        data = np.fliplr(np.frombuffer(fbo.read(components=4), dtype=np.dtype('u1')).reshape((out_shape[1], out_shape[0], 4)))

        # release memory
        fbo.release()
        rbo.release()
        dbo.release()
        vbo.release()
        vao.release()

        return data
