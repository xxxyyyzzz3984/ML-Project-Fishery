import random

import numpy
import copy

import tensorflow as tf

scan_wnd_size = [100, 100]

save_models_dir = '../../saved models/'


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

W_conv4 = weight_variable([3, 3, 64, 128], name='wconv4_pnet')
b_conv4 = bias_variable([128], name='bconv4_pnet')
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([3, 3, 128, 128], name='wconv5_pnet')
b_conv5 = bias_variable([128], name='bconv5_pnet')
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([3, 3, 128, 256], name='wconv5_pnet')
b_conv6 = bias_variable([256], name='bconv5_pnet')
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)

## fully connected
W_fc1 = weight_variable([3 * 3 * 256, 256], name='wfc1_pnet')
b_fc1 = bias_variable([256], name='bfc1_pnet')

h_pool2_flat = tf.reshape(h_pool6, [-1, 3 * 3 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='hfc1drop_pnet')

W_fc2 = weight_variable([256, 2], name='wfc2_pnet')
b_fc2 = bias_variable([2], name='bfc2_pnet')


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) +
    tf.nn.softmax_cross_entropy_with_logits(labels=1-y_, logits=1-y_conv))


mse = tf.reduce_mean(tf.square(y_-y_conv))

loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)),
                                            reduction_indices=1)))

# loss = tf.reduce_mean(tf.square(tf.norm(tf.subtract(y_, y_conv)), 2))

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



saver = tf.train.Saver()
#
# saver.restore(sess, save_models_dir + 'onet_train.ckpt')
# print("Model restored.")


min_acc = 100
max_acc = 0
total_acc = 0
file_count = 1

while True:

    y_data_list = []

    x_data_fish = numpy.load('../../array train dataset/big_pic_blocks/fish_parts_100x100/it%d_image_%dx%d.npy'
                             % (file_count, scan_wnd_size[0], scan_wnd_size[1]))

    x_data_nofish = numpy.load('../../array train dataset/big_pic_blocks/nofish_parts_100x100/it%d_image_%dx%d.npy'
                               % (file_count, scan_wnd_size[0], scan_wnd_size[1]))

    x_data = numpy.concatenate((x_data_fish, x_data_nofish))

    for i in range(x_data_fish.shape[0]):
        y_data_list.append([1, 0])

    for j in range(x_data_nofish.shape[0]):
        y_data_list.append([0, 1])

    y_data = numpy.array(y_data_list)

    ####shuffle
    for i in range(x_data.shape[0] / 2):
        if i % 2 == 0:
            j = x_data.shape[0]

            x_tmp = copy.copy(x_data[i])
            x_data[i] = copy.copy(x_data[j - i - 1])
            x_data[j - i - 1] = copy.copy(x_tmp)

            y_tmp = copy.copy(y_data[i])
            y_data[i] = copy.copy(y_data[j - i - 1])
            y_data[j - i - 1] = copy.copy(y_tmp)
    #########


    train_step.run({y_: y_data, x: x_data, keep_prob: 0.5}, sess)

    e = sess.run(cross_entropy, feed_dict={y_: y_data, x: x_data, keep_prob: 1.0})
    train_accuracy = accuracy.eval(feed_dict={y_: y_data, x: x_data, keep_prob: 1.0})

    print train_accuracy
    print e
    print

    if min_acc > train_accuracy:
        min_acc = train_accuracy

    if max_acc < train_accuracy:
        max_acc = train_accuracy

    total_acc += train_accuracy

    file_count += 1


    if file_count > 188:

        print 'save model'
        print 'minimum acc is %f' % min_acc
        print 'maximum acc is %f' % max_acc
        print 'average acc is %f' % (total_acc / 188)

        save_path = saver.save(sess, save_path=save_models_dir + 'onet_train.ckpt')
        file_count = 1

        if min_acc > 0.99:
            print 'break'
            break

        min_acc = 100