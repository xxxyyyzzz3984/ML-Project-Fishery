from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy
import copy

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
scan_wnd_size = [48, 48]

x = tf.placeholder(tf.float32, shape=[None, scan_wnd_size[0] * scan_wnd_size[1], 3])
y_ = tf.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 3], name='image_pnet')

W_conv1 = weight_variable([3, 3, 3, 32], name='wconv1_pnet')
b_conv1 = bias_variable([32], name='bconv1_pnet')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_3x3(h_conv1)  ## one layer 3x3 max pooling

W_conv2 = weight_variable([3, 3, 32, 64], name='wconv2_pnet')
b_conv2 = bias_variable([64], name='bconv2_pnet')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_3x3(h_conv2)  ## one layer 2x2 max pooling

W_conv3 = weight_variable([3, 3, 64, 64], name='wconv3_pnet')
b_conv3 = bias_variable([64], name='bconv3_pnet')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)  ## one layer 2x2 max pooling

W_conv4 = weight_variable([2, 2, 64, 128], name='wconv4_pnet')
b_conv4 = bias_variable([128], name='bconv4_pnet')
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

## fully connected
W_fc1 = weight_variable([3 * 3 * 128, 256], name='wfc1_pnet')
b_fc1 = bias_variable([256], name='bfc1_pnet')

h_pool2_flat = tf.reshape(h_conv4, [-1, 3 * 3 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([256, 2], name='wfc2_pnet')
b_fc2 = bias_variable([2], name='bfc2_pnet')


y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
test_pics = [f for f in listdir('../test dataset/')
               if isfile(join('../test dataset/', f))]


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_models_dir + 'onet_train.ckpt')
    print("Model restored.")

    for test_pic_name in test_pics:
        test_pic_path = '../test dataset/' + test_pic_name

        image = io.imread(test_pic_path)
        image_data = transform.resize(image, numpy.array(scan_wnd_size))
        image_data = numpy.array(image_data, dtype=float)
        image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1], 3)
        y_predict = y_conv.eval({x: image_data}, sess)

        print y_predict

        # search_stride = 50
        #
        # i_w = 200
        # j_w = 200
        #
        # i_total = image.shape[0] / search_stride
        # j_total = image.shape[1] / search_stride
        #
        # plt.imshow(image)
        # ax = plt.gca()
        #
        # image_grey = color.rgb2grey(image)
        # edges_data = canny(image_grey)
        # image_grey = filters.gaussian(edges_data, 2)
        #
        # for i in range(i_total):
        #     for j in range(j_total):
        #
        #         image_data = copy.copy(image_grey[i * search_stride:i * search_stride + i_w, j * search_stride:j * search_stride + j_w])
        #
        #
        #         image_data = transform.resize(image_data, numpy.array(scan_wnd_size))
        #         image_data = numpy.array(image_data, dtype=float)
        #         image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1])
        #         y_predict = y_conv.eval({x: image_data}, sess)
        #
        #         if y_predict[0][0] > y_predict[0][1]:
        #             rect = mpatches.Rectangle((j * search_stride, i * search_stride), j_w, i_w,
        #                                       fill=False, edgecolor='red', linewidth=2)
        #
        #             ax.add_patch(rect)
        #
        # plt.show()




    # count = 0
    # for i in range(100):
    #     y_predict = y_conv.eval({x: image_data}, sess)
    #     if y_predict[0][0] > y_predict[0][1]:
    #         count += 1

    # y_predict = y_conv.eval({x: image_data}, sess)
    #
    # print y_predict

# ## PNET Part Finishes
# ##############