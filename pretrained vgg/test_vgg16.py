import argparse
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from codes.vgg import vgg16
from codes.vgg import utils
import tensorflow as tf
from skimage import transform, io
import copy
from os import listdir
from os.path import isfile, join
import numpy

img1 = utils.load_image("../../small test/2.jpg")

batch1 = img1.reshape((1, 224, 224, 3))


test_pics_folder = '../../test dataset/'
test_pics = [f for f in listdir(test_pics_folder)
               if isfile(join(test_pics_folder, f))]

img_count = 0

ap = argparse.ArgumentParser()
ap.add_argument('-i', nargs=1)
ap.add_argument('-si', nargs=1)
ap.add_argument('-sj', nargs=1)
ap.add_argument('-ei', nargs=1)
ap.add_argument('-ej', nargs=1)
ap.add_argument('-w', nargs=1)
ap.add_argument('-ss', nargs=1)

opts = ap.parse_args()
image_filepath = opts.i[0]
start_i = int(opts.si[0])
end_i = int(opts.ei[0])
start_j = int(opts.sj[0])
end_j = int(opts.ej[0])
scan_width = int(opts.w[0])
search_stride = int(opts.ss[0])


with tf.Session() as sess:
    images = tf.placeholder("float", [1, 224, 224, 3])
    feed_dict = {images: batch1}
    img_cand = []
    img_coor = []


    image = io.imread(image_filepath)
    # search_stride = 100

    i_w = scan_width
    j_w = scan_width

    i_total = image.shape[0] / search_stride
    j_total = image.shape[1] / search_stride

    image_list = []

    for i in range(start_i, end_i):
        for j in range(start_j, end_j):

            image_data = copy.copy(image[i * search_stride:i * search_stride + i_w,
                                   j * search_stride:j * search_stride + j_w, 0:3])

            try:
                image_data = transform.resize(image_data, [224, 224])
                image_data = image_data.reshape([224, 224, 3])
                image_data = numpy.array(image_data, dtype=numpy.float32)
                image_list.append(image_data)
            except:
                continue

    if len(image_list) > 0:
        vgg = vgg16.Vgg16(vgg16_npy_path='../../saved models/vgg16.npy')
        image_array = numpy.array(image_list)
        image_list = []

        with tf.name_scope("content_vgg"):
            vgg.build(image_array)

        probs = sess.run(vgg.prob, feed_dict=feed_dict)

        for i in range(probs.shape[0]):
            result = utils.print_prob(probs[i], '../../saved models/synset.txt')
            print result

    else:
        print ''

