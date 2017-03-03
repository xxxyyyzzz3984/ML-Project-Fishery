import numpy
import copy

import tensorflow as tf

scan_wnd_size = [256, 256]

save_models_dir = '../saved models/'


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def conv2d_a(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')

####load data ###################
# y_data_list = []
# x_data = numpy.load('../array train dataset/fish_wholeimg_%dx%d.npy' % (scan_wnd_size[0],
#                           scan_wnd_size[1]))
#
#
# y_data = numpy.load('../array train dataset/fishlabels_wholeimg_%dx%d.npy' %
#                                          (scan_wnd_size[0], scan_wnd_size[1]))


# print x_data.shape
# print y_data.shape

####shuffle
# for i in range(x_data.shape[0]/2):
#     if i % 2 == 0:
#         j = x_data.shape[0]
#
#         x_tmp = copy.copy(x_data[i])
#         x_data[i] = copy.copy(x_data[j-i-1])
#         x_data[j-i-1] = copy.copy(x_tmp)
#
#         y_tmp = copy.copy(y_data[i])
#         y_data[i] = copy.copy(y_data[j - i - 1])
#         y_data[j - i - 1] = copy.copy(y_tmp)
#########

#########################

x = tf.placeholder(tf.float32, shape=[None, scan_wnd_size[0] * scan_wnd_size[1]])
y_ = tf.placeholder(tf.float32, [None, 4])

x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 1], name='image_pnet')

W_conv1 = weight_variable([3, 3, 1, 16], name='wconv1_pnet')
b_conv1 = bias_variable([16], name='bconv1_pnet')
h_conv1 = tf.nn.sigmoid(conv2d_a(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  ## one layer 3x3 max pooling
# h_norm1 = tf.contrib.layers.batch_norm(h_pool1)

W_conv2 = weight_variable([3, 3, 16, 32], name='wconv2_pnet')
b_conv2 = bias_variable([32], name='bconv2_pnet')
h_conv2 = tf.nn.sigmoid(conv2d_a(h_pool1, W_conv2) + b_conv2)
# h_norm2 = tf.contrib.layers.batch_norm(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)  ## one layer 2x2 max pooling

W_conv3 = weight_variable([3, 3, 32, 64], name='wconv3_pnet')
b_conv3 = bias_variable([64], name='bconv3_pnet')
h_conv3 = tf.nn.sigmoid(conv2d_a(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)  ## one layer 2x2 max pooling

W_conv4 = weight_variable([3, 3, 64, 128], name='wconv4_pnet')
b_conv4 = bias_variable([128], name='bconv4_pnet')
h_conv4 = tf.nn.sigmoid(conv2d_a(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([3, 3, 128, 128], name='wconv5_pnet')
b_conv5 = bias_variable([128], name='bconv5_pnet')
h_conv5 = tf.nn.sigmoid(conv2d_a(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

W_conv6 = weight_variable([3, 3, 128, 256], name='wconv6_pnet')
b_conv6 = bias_variable([256], name='bconv6_pnet')
h_conv6 = tf.nn.sigmoid(conv2d_a(h_pool5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)

W_conv7 = weight_variable([1, 1, 256, 256], name='wconv7_pnet')
b_conv7 = bias_variable([256], name='bconv7_pnet')
h_conv7 = tf.nn.relu(conv2d_a(h_pool6, W_conv7) + b_conv7)
h_pool7 = max_pool_2x2(h_conv7)


## fully connected
W_fc1 = weight_variable([2*2*256, 256], name='wfc1_pnet')
b_fc1 = bias_variable([256], name='bfc1_pnet')

h_pool2_flat = tf.reshape(h_pool7, [-1, 2*2*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='hfc1drop_pnet')


W_fc2 = weight_variable([256, 4], name='wfc2_pnet')
b_fc2 = bias_variable([4], name='bfc2_pnet')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# x = tf.placeholder(tf.float32, shape=[None, scan_wnd_size[0] * scan_wnd_size[1]])
# y_ = tf.placeholder(tf.float32, [None, 4])
#
# x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 1], name='image_pnet')
#
# W_conv1 = weight_variable([3, 3, 1, 32], name='wconv1_pnet')
# b_conv1 = bias_variable([32], name='bconv1_pnet')
# h_conv1 = tf.nn.relu(conv2d_a(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_3x3(h_conv1)  ## one layer 3x3 max pooling
#
# W_conv2 = weight_variable([3, 3, 32, 64], name='wconv2_pnet')
# b_conv2 = bias_variable([64], name='bconv2_pnet')
# h_conv2 = tf.nn.relu(conv2d_a(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_3x3(h_conv2)  ## one layer 2x2 max pooling
#
# W_conv3 = weight_variable([3, 3, 64, 64], name='wconv3_pnet')
# b_conv3 = bias_variable([64], name='bconv3_pnet')
# h_conv3 = tf.nn.relu(conv2d_a(h_pool2, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)  ## one layer 2x2 max pooling
#
# W_conv4 = weight_variable([2, 2, 64, 128], name='wconv4_pnet')
# b_conv4 = bias_variable([128], name='bconv4_pnet')
# h_conv4 = tf.nn.relu(conv2d_a(h_pool3, W_conv4) + b_conv4)
#
# ## fully connected
# W_fc1 = weight_variable([15 * 15 * 128, 256], name='wfc1_pnet')
# b_fc1 = bias_variable([256], name='bfc1_pnet')
#
# h_pool2_flat = tf.reshape(h_conv4, [-1, 15 * 15 * 128])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='hfc1drop_pnet')
#
# W_fc2 = weight_variable([256, 4], name='wfc2_pnet')
# b_fc2 = bias_variable([4], name='bfc2_pnet')
#
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


mse = tf.reduce_mean(tf.square(y_-y_conv))

loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)),
                                            reduction_indices=1)))

# loss = tf.reduce_mean(tf.square(tf.norm(tf.subtract(y_, y_conv)), 2))

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#
saver.restore(sess, save_models_dir + 'onet_train.ckpt')
print("Model restored.")




step = 1
min_e = 27856424.0
max_e = 0.0
total_acc = 0
error_sum = 0
while True:

    y_data_list = []
    x_data = numpy.load('../array train dataset/big_pic_blocks/it%d_image_%dx%d.npy'
                        % (step, scan_wnd_size[0], scan_wnd_size[1]))

    y_data = numpy.load('../array train dataset/big_pic_blocks/it%d_labels_%dx%d.npy'
                        % (step, scan_wnd_size[0], scan_wnd_size[1]))

    if step == 67:

        print 'minimum loss is %f' % min_e
        print 'maximum loss is %f' % max_e
        # print 'average accuracy is %f' % avg_acc
        print 'average loss is %f' % (float(error_sum)/step)
        print 'Save model'
        print

        save_path = saver.save(sess, save_path=save_models_dir + 'onet_train.ckpt')

        if max_e < 1:
            print 'Break the training loop...'
            save_path = saver.save(sess, save_path=save_models_dir + 'onet_train.ckpt')
            break

        else:
            batch_i = 0
            error_sum = 0
            min_e = 27856424
            max_e = 0

        step = 0


    train_step.run({y_: y_data, x: x_data, keep_prob: 0.5}, sess)

    e = sess.run(loss, feed_dict={y_: y_data, x: x_data, keep_prob: 1.0})
    train_accuracy = accuracy.eval(feed_dict={y_: y_data, x: x_data, keep_prob: 1.0})

    error_sum += e

    if min_e > e:
        min_e = e

    if max_e < e:
        max_e = e

    # if step % 100 == 0:
    #     print 'save model'
    #     save_path = saver.save(sess, save_path=save_models_dir + 'onet_train.ckpt')
    #     step = 0

    print e

    step += 1
