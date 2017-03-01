import tensorflow as tf
import numpy
import copy

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
# ## ONET Part
# ##############
scan_wnd_size = [48, 48]
y_data_list = []
x_data = numpy.load('../array train dataset/fish_imagedata_%dx%d.npy' % (scan_wnd_size[0],
                                                                         scan_wnd_size[1]))
fish_len = x_data.shape[0]

x_data = numpy.append(x_data, numpy.load('../array train dataset/nofish_imagedata_%dx%d.npy' %
                                         (scan_wnd_size[0], scan_wnd_size[1])))

total_len = x_data.shape[0]/(scan_wnd_size[0] * scan_wnd_size[1])

for i in range(total_len):
    if i < fish_len:
        y_data_list.append([1, 0]) ## [0, 1] denotes has fish
    else:
        y_data_list.append([0, 1]) ## [1, 0] denotes no fish

x_data = x_data.reshape(total_len, scan_wnd_size[0] * scan_wnd_size[1])
y_data = numpy.array(y_data_list)
y_data = y_data.reshape(len(y_data), 2)

####shuffle
for i in range(x_data.shape[0]/2):
    if i % 2 == 0:
        j = x_data.shape[0]

        x_tmp = copy.copy(x_data[i])
        x_data[i] = copy.copy(x_data[j-i-1])
        x_data[j-i-1] = copy.copy(x_tmp)

        y_tmp = copy.copy(y_data[i])
        y_data[i] = copy.copy(y_data[j - i - 1])
        y_data[j - i - 1] = copy.copy(y_tmp)
#########

x = tf.placeholder(tf.float32, shape=[None, scan_wnd_size[0] * scan_wnd_size[1]])
y_ = tf.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 1], name='image_pnet')

W_conv1 = weight_variable([5, 5, 1, 32], name='wconv1_pnet')
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


W_fc1 = weight_variable([3*3*128, scan_wnd_size[0] * scan_wnd_size[1]], name='wfc1_pnet')
b_fc1 = bias_variable([scan_wnd_size[0] * scan_wnd_size[1]], name='bfc1_pnet')

h_pool2_flat = tf.reshape(h_conv4, [-1, 3*3*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([scan_wnd_size[0] * scan_wnd_size[1], 2], name='wfc2_pnet')
b_fc2 = bias_variable([2], name='bfc2_pnet')
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

y_conv = h_fc2


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    -tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) -
    tf.nn.softmax_cross_entropy_with_logits(labels=1-y_, logits=1-y_conv))


mse = tf.reduce_mean(tf.square(y_-y_conv))


correct_prediction = tf.equal(y_conv, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#
# saver.restore(sess, save_models_dir + 'onet_train.ckpt')
# print("Model restored.")

step = 0
while True:
    train_step.run({y_: y_data, x: x_data}, sess)
    e = sess.run(cross_entropy, feed_dict={y_: y_data, x: x_data})
    train_accuracy = accuracy.eval(feed_dict={y_: y_data, x: x_data})
    print 'accuracy is %f' % train_accuracy
    print e
    if train_accuracy > 0.98:
        print 'Break the training loop...'
        save_path = saver.save(sess, save_path=save_models_dir + 'onet_train.ckpt')
        break

    if step % 500 == 0:
        print 'saving the model'
        save_path = saver.save(sess, save_path=save_models_dir + 'onet_train.ckpt')

    step += 1

# ##############
# ## ONET Part Finishes
# ##############