import tensorflow as tf
import numpy

'''Inspired by CVPR 16 paper, Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks'''


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# ##############
# ## PNET Part
# ##############
scan_wnd_size = [12, 12]
y_data_list = []
x_data = numpy.load('../array train dataset/fish_imagedata_%dx%d.npy' % (scan_wnd_size[0],
                                                                         scan_wnd_size[1]))
fish_len = x_data.shape[0]

x_data = numpy.append(x_data, numpy.load('../array train dataset/nofish_imagedata_%dx%d.npy' %
                                         (scan_wnd_size[0], scan_wnd_size[1])))

total_len = x_data.shape[0]/(scan_wnd_size[0] * scan_wnd_size[1])

for i in range(total_len):
    if i < fish_len:
        y_data_list.append([0, 1]) ## [0, 1] denotes has fish
    else:
        y_data_list.append([1, 0]) ## [1, 0] denotes no fish

x_data = x_data.reshape(total_len, scan_wnd_size[0] * scan_wnd_size[1])
y_data = numpy.array(y_data_list)
y_data = y_data.reshape(len(y_data), 2)

x = tf.placeholder(tf.float32, shape=[None, scan_wnd_size[0] * scan_wnd_size[1]])
y_ = tf.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 1])

W_conv1 = weight_variable([3, 3, 1, 10])
b_conv1 = bias_variable([10])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  ## one layer 2x2 max pooling

W_conv2 = weight_variable([3, 3, 10, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  ## one layer 2x2 max pooling

W_conv3 = weight_variable([3, 3, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# W_conv4 = weight_variable([1, 1, 32, 1])
# b_conv4 = bias_variable([1])
# h_conv4 = tf.nn.softmax(conv2d(h_conv3, W_conv4) + b_conv4)

W_fc1 = weight_variable([3*3*32, scan_wnd_size[0] * scan_wnd_size[1]])
b_fc1 = bias_variable([scan_wnd_size[0] * scan_wnd_size[1]])

h_pool2_flat = tf.reshape(h_conv3, [-1, 3*3*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([scan_wnd_size[0] * scan_wnd_size[1], 2])
b_fc2 = bias_variable([2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

y_conv = h_fc2


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) + \
    (tf.nn.softmax_cross_entropy_with_logits(labels=1-y_, logits=1-y_conv)))

mse = tf.reduce_mean(tf.square(y_-y_conv))


train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())

while True:
  train_step.run({y_: y_data, x: x_data}, sess)
  e = sess.run(cross_entropy, feed_dict={y_: y_data, x: x_data})
  print e
  if e < 1e-6:
      print e
      break

# ##############
# ## PNET Part Finishes
# ##############