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
ap.add_argument('-x', nargs=1)
ap.add_argument('-y', nargs=1)
ap.add_argument('-wr', nargs=1)
ap.add_argument('-hr', nargs=1)

opts = ap.parse_args()
image_filepath = opts.i[0]
x = int(opts.x[0])
y = int(opts.y[0])
w = int(opts.wr[0])
h = int(opts.hr[0])


with tf.Session() as sess:
    images = tf.placeholder("float", [1, 224, 224, 3])

    image = io.imread(image_filepath)


    image_data = copy.copy(image[y: y + h, x:x + w, 0:3])
    image_data = transform.resize(image_data, [224, 224])
    image_data = image_data.reshape([1, 224, 224, 3])
    image_data = numpy.array(image_data, dtype=numpy.float32)


    vgg = vgg16.Vgg16(vgg16_npy_path='../saved models/vgg16.npy')

    with tf.name_scope("content_vgg"):
        vgg.build(image_data)

    probs = sess.run(vgg.prob, feed_dict={images: image_data})

    result = utils.print_prob(probs[0], '../saved models/synset.txt')
    print result