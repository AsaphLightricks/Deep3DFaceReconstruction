import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import traceback
import glob
from scipy.io import loadmat, savemat
import imageio as iio
import matplotlib.pyplot as plt

from preprocess_img import Preprocess
from load_data import *
from reconstruct_mesh import Reconstruction

from rendering import Renderer


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


import dlib

det = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
renderer = Renderer()


def get_landmarks(image):
    detection = det(image, 1)[0]
    face_shape = pred(image, detection)

    l_eye_x = np.mean([face_shape.part(i).x for i in range(42, 48)])
    l_eye_y = np.mean([face_shape.part(i).y for i in range(42, 48)])

    r_eye_x = np.mean([face_shape.part(i).x for i in range(36, 42)])
    r_eye_y = np.mean([face_shape.part(i).y for i in range(36, 42)])

    l_eye = (l_eye_x, l_eye_y)
    r_eye = (r_eye_x, r_eye_y)

    eyes = np.vstack((l_eye, r_eye))

    nose = face_shape.part(30)
    l_mouth = face_shape.part(48)
    r_mouth = face_shape.part(54)

    pp = [(p.x, p.y) for p in [nose, r_mouth, l_mouth]]

    return np.vstack((eyes, pp))


def demo():
    # input and output folder
    in_dir = 'input_vids'
    out_dir = 'output2'
    # img_list = glob.glob(image_path + '/' + '*.jpg')
    vid_list = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if not f.startswith('.')]

    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile('./BFM/BFM_model_front.mat'):
        transferBFM09()

    # read face model
    facemodel = BFM()
    # read standard landmarks for preprocessing images
    lm3D = load_lm3d()
    n = 0

    # build reconstruction model
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):

        images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
        graph_def = load_graph('network/FaceReconModel.pb')
        tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

        # output coefficients of R-Net (dim = 257)
        coeff = graph.get_tensor_by_name('resnet/coeff:0')

        with tf.Session() as sess:
            print('reconstructing...')
            for file in vid_list:
                print(file)
                with iio.get_reader(file) as reader:
                    fps = reader.get_meta_data()['fps']
                    name, ext = os.path.splitext(file)
                    file_name = os.path.basename(name)
                    l_writer = iio.get_writer(os.path.join(out_dir, file_name + ext), fps=fps)
                    # r_writer = iio.get_writer(os.path.join(out_dir, file_name + '_render' + ext), fps=fps)
                    for i, im in enumerate(reader):
                        print(i)
                        try:
                            # load images and corresponding 5 facial landmarks
                            # img,lm = load_img(file,file.replace('png','txt'))
                            img = Image.fromarray(im)
                            np_img = np.array(img)
                            lm = get_landmarks(np_img)
                            h, w = np_img.shape[:2]

                            # preprocess input image
                            input_img, lm_new, transform_params = Preprocess(img, lm, lm3D)
                            s = transform_params[2]
                            out_sh = int(np.round(224 / s))
                            out_sh = min(out_sh, min(w, h))

                            coef = sess.run(coeff, feed_dict={images: input_img})

                            # reconstruct 3D face with output coefficients and face model
                            face_shape, face_texture, face_color, tri, face_projection, z_buffer, landmarks_2d, translation, rotation, projection = Reconstruction(
                                coef, facemodel)

                            # reshape outputs
                            input_img = np.squeeze(input_img)
                            shape = np.squeeze(face_shape, (0))
                            color = np.squeeze(face_color, (0))
                            landmarks_2d = np.squeeze(landmarks_2d, (0))

                            cx, cy = transform_params[3][0], transform_params[4][0]
                            tx, ty = -(w / 2 - cx), -(cy - h / 2)

                            land_im = np_img.copy()
                            for x, y in landmarks_2d:
                                x = int(np.round((x + (w * s - 224) // 2) / s + tx))
                                y = int(np.round((y + (h * s - 224) // 2) / s + ty))
                                cv2.circle(land_im, (x, y), 2, (0, 255, 0), -1)

                            trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])

                            # plt.imshow(land_im)
                            # plt.show()
                            rendered = renderer.render(shape, color / 255, np.squeeze(tri.astype(np.int) - 1),
                                                       projection, rotation, translation, (out_sh, out_sh))

                            out = np.zeros((h, w, 4), dtype=np_img.dtype)
                            oo = out_sh // 2
                            print(out_sh, oo, rendered.shape, out.shape)
                            out[h // 2 - oo: h // 2 + oo + out_sh % 2, w // 2 - oo: w // 2 + oo + out_sh % 2, :] = rendered
                            # plt.imshow(out)
                            # plt.show()
                            im_trans = cv2.warpAffine(out, trans_mat, (w, h))

                            alpha = (im_trans[..., 3] / 255).astype(np.uint8)
                            rendered = im_trans[..., :3] * alpha[..., np.newaxis] + np_img * (1 - alpha[..., None])

                            out_im = np.hstack([np_img, rendered, land_im])
                            l_writer.append_data(out_im)
                            # plt.imshow(rendered)
                            # plt.show()

                            # mesh_im = im.copy()
                            # for x, y in face_projection.squeeze()[::20]:
                            # 	x = int(np.round(x))
                            # 	y = int(np.round(y))
                            # 	cv2.circle(mesh_im, (x, y), 1, (255, 255, 0), -1)

                            # plt.imshow(mesh_im)
                            # plt.show()

                            # save output files
                            # cropped image, which is the direct input to our R-Net
                            # 257 dim output coefficients by R-Net
                            # 68 face landmarks of cropped image
                            # savemat(os.path.join(save_path,os.path.basename(file).replace('.jpg','.mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coef,'landmarks_2d':landmarks_2d,'lm_5p':lm_new})
                            # save_obj(os.path.join(save_path,os.path.basename(file).replace('.jpg','_mesh.obj')),shape,tri,np.clip(color,0,255)/255) # 3D reconstruction face (in canonical view)
                        except Exception as e:
                            l_writer.append_data(np.hstack([im] * 3))
                            print(traceback.print_exc())


if __name__ == '__main__':
    demo()
