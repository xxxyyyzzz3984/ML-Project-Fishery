import copy
import random

import tensorflow as tf
import numpy
from os import listdir

from os.path import isfile, join
from skimage import io, transform, filters, color
from skimage.feature import canny
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


'''Inspired by CVPR 16 paper, Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks'''

save_models_dir = '../saved models/'


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')


# ##############
# ## PNET Part
# ##############
scan_wnd_size = [12, 12]

x = tf.placeholder(tf.float32, shape=[None, scan_wnd_size[0] * scan_wnd_size[1], 3])
y_ = tf.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 3], name='image_pnet')

W_conv1 = weight_variable([5, 5, 3, 10], name='wconv1_pnet')
b_conv1 = bias_variable([10], name='bconv1_pnet')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  ## one layer 2x2 max pooling

W_conv2 = weight_variable([5, 5, 10, 16], name='wconv2_pnet')
b_conv2 = bias_variable([16], name='bconv2_pnet')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv3 = weight_variable([5, 5, 16, 32], name='wconv3_pnet')
b_conv3 = bias_variable([32], name='bconv3_pnet')
h_conv3 = tf.nn.softmax(conv2d(h_conv2, W_conv3) + b_conv3)

W_fc1 = weight_variable([6 * 6 * 32, 1024], name='wfc1_pnet')
b_fc1 = bias_variable([1024], name='bfc1_pnet')

h_pool2_flat = tf.reshape(h_conv3, [-1, 6 * 6 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([1024, 2], name='wfc2_pnet')
b_fc2 = bias_variable([2], name='bfc2_pnet')

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


test_pics_folder = '../test dataset/'

test_pics = [f for f in listdir(test_pics_folder)
               if isfile(join(test_pics_folder, f))]


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_models_dir + 'pnet_model1_incomplete/pnet_train.ckpt')
    print("Model restored.")

    for test_pic_name in test_pics:
        test_pic_path = test_pics_folder + test_pic_name
    #
        image = io.imread(test_pic_path)

        image = transform.rescale(image, 0.6)
    #     image_data = transform.resize(image, numpy.array(scan_wnd_size))
    #     image_data = numpy.array(image_data, dtype=float)
    #     image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1], 3)
    #     y_predict = y_conv.eval({x: image_data}, sess)
    #
    #     plt.imshow(image)
    #     ax = plt.gca()
    #     rect = mpatches.Rectangle((y_predict[0][0], y_predict[0][1]),
    #                               y_predict[0][2], y_predict[0][3],
    #                               fill=False, edgecolor='red', linewidth=2)
    #
    #     ax.add_patch(rect)
    #     plt.show()



        search_stride = 50

        i_w = 60
        j_w = 60

        i_total = image.shape[0] / search_stride
        j_total = image.shape[1] / search_stride

        plt.imshow(image)
        ax = plt.gca()

        image_grey = color.rgb2grey(image)

        for i in range(i_total):
            for j in range(j_total):

                # image_data = copy.copy(image_grey[i * search_stride:i * search_stride + i_w, j * search_stride:j * search_stride + j_w])

                image_data = transform.resize(image, numpy.array(scan_wnd_size))
                image_data = numpy.array(image_data, dtype=float)
                image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1], 3)
                y_predict = y_conv.eval({x: image_data}, sess)
                print y_predict

                if y_predict[0][0] > y_predict[0][1] + 0.6:
                    print y_predict
                    rect = mpatches.Rectangle((j * search_stride, i * search_stride), j_w, i_w,
                                              fill=False, edgecolor='red', linewidth=2)

                    ax.add_patch(rect)

        plt.show()