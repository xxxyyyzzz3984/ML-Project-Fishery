import tensorflow as tf
import numpy

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

x_image = tf.reshape(x, [-1, scan_wnd_size[0], scan_wnd_size[1], 1], name='image_pnet')

W_conv1 = weight_variable([3, 3, 1, 10], name='wconv1_pnet')
b_conv1 = bias_variable([10], name='bconv1_pnet')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  ## one layer 2x2 max pooling

W_conv2 = weight_variable([3, 3, 10, 16], name='wconv2_pnet')
b_conv2 = bias_variable([16], name='bconv2_pnet')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  ## one layer 2x2 max pooling

W_conv3 = weight_variable([3, 3, 16, 32], name='wconv3_pnet')
b_conv3 = bias_variable([32], name='bconv3_pnet')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_fc1 = weight_variable([3*3*32, scan_wnd_size[0] * scan_wnd_size[1]], name='wfc1_pnet')
b_fc1 = bias_variable([scan_wnd_size[0] * scan_wnd_size[1]], name='bfc1_pnet')

h_pool2_flat = tf.reshape(h_conv3, [-1, 3*3*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([scan_wnd_size[0] * scan_wnd_size[1], 2], name='wfc2_pnet')
b_fc2 = bias_variable([2], name='bfc2_pnet')
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

y_conv = h_fc2


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

mse = tf.reduce_mean(tf.square(y_-y_conv))


train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

step = 0
while True:
    train_step.run({y_: y_data, x: x_data}, sess)
    e = sess.run(cross_entropy, feed_dict={y_: y_data, x: x_data})
    print e
    if e < 1e-4:
        print 'Break the training loop...'
        save_path = saver.save(sess, save_path=save_models_dir + 'pnet_train.ckpt')
        break

    if step % 500 == 0:
        print 'saving the model'
        save_path = saver.save(sess, save_path=save_models_dir + 'pnet_train.ckpt')

    step += 1

# ##############
# ## PNET Part Finishes
# ##############