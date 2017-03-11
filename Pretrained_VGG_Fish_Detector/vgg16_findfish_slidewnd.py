import argparse
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from codes.vgg import vgg16
from codes.vgg import utils
import tensorflow as tf
from skimage import transform, io
import copy
import numpy

ap = argparse.ArgumentParser()
ap.add_argument('-i', nargs=1)
ap.add_argument('-ti', nargs=1)
ap.add_argument('-ss', nargs=1)
ap.add_argument('-w', nargs=1)

opts = ap.parse_args()
image_filepath = opts.i[0]
search_stride = int(opts.ss[0])
target_i = int(opts.ti[0])
width = int(opts.w[0])


with tf.Session() as sess:

    image = io.imread(image_filepath)

    image_list = []

    i_w = width
    j_w = width

    i_total = image.shape[0] / search_stride
    j_total = image.shape[1] / search_stride
    candidates = []

    for j in range(j_total):
        image_data = copy.copy(image[j*search_stride:j*search_stride+j_w,
                               target_i*search_stride:target_i*search_stride + i_w, 0:3])

        try:
            image_data = transform.resize(image_data, [224, 224])
            image_data = image_data.reshape([224, 224, 3])
            image_data = numpy.array(image_data, dtype=numpy.float32)
            image_list.append(image_data)
        except:
            continue

    image_arr = numpy.array(image_list)

    images = tf.placeholder("float", [image_arr.shape[0], 224, 224, 3])


    vgg = vgg16.Vgg16(vgg16_npy_path='../saved models/vgg16.npy')

    with tf.name_scope("content_vgg"):
        vgg.build(image_arr)

    probs = sess.run(vgg.prob, feed_dict={images: image_arr})

    for i in range(probs.shape[0]):
        result = utils.print_prob(probs[i], '../saved models/synset.txt')
        print result