import ImageFilter

import tensorflow as tf
import numpy
import Image
from resizeimage import resizeimage

import copy
from PIL import ImageFilter, Image
import pickle


import numpy
from resizeimage import resizeimage
from os import listdir
from os.path import isfile, join
from skimage.morphology import watershed
from skimage.filters import sobel
from skimage.feature import hog
from skimage import data, color, exposure, filters, io, transform
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy import ndimage as ndi

from skimage.segmentation import random_walker


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

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='hfc1drop_pnet')

W_fc2 = weight_variable([1024, 2], name='wfc2_pnet')
b_fc2 = bias_variable([2], name='bfc2_pnet')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

mse = tf.reduce_mean(tf.square(y_-y_conv))


train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, save_models_dir + 'pnet_train.ckpt')
print("Model restored.")

image_data = io.imread('../small test/2.png')


# edges_data = canny(image_data)

image_data = transform.resize(image_data, numpy.array(scan_wnd_size))

image_data = numpy.array(image_data, dtype=float)

image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1], 3)


y_predict = y_conv.eval({x: image_data, keep_prob: 0.5}, sess)
print y_predict


# ## PNET Part Finishes
# ##############