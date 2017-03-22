import random

import numpy
import tensorflow as tf
import copy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding='SAME')

def conv2d_reduce(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2_reduce(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3_reduce(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

#########loading data##########
target_wnd_size = [220, 220]
file_len = 235


data_dir = '../../array train dataset/fish types/rotated_%dx%d/' % (target_wnd_size[0], target_wnd_size[1])



##############
## ONET
##############
# x = tf.placeholder("float", shape=[None, 48, 48, 3])
# y_ = tf.placeholder("float", shape=[None, 7])
# x_image = tf.reshape(x, [-1, 48, 48, 3])
#
# # convolutional layers
# W_conv1 = weight_variable([3, 3, 3, 32])
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_3x3_reduce(h_conv1)  ## one layer 3x3 max pooling
#
# W_conv2 = weight_variable([3, 3, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_3x3_reduce(h_conv2)  ## one layer 2x2 max pooling
#
# W_conv3 = weight_variable([3, 3, 64, 64])
# b_conv3 = bias_variable([64])
# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2_reduce(h_conv3)  ## one layer 2x2 max pooling
#
# W_conv4 = weight_variable([3, 3, 64, 128])
# b_conv4 = bias_variable([128])
# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#
#
# # fully connected layer
# W_fc1 = weight_variable([6 * 6 * 128, 256])
# b_fc1 = bias_variable([256])
#
# h_pool2_flat = tf.reshape(h_conv4, [-1, 6 * 6 * 128])  # flat into 1 dimention
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# # dropout
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # kill some neuron
#
# # Readout Layer
# W_fc2 = weight_variable([256, 7])
# b_fc2 = bias_variable([7])
#
# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#######################

##################
## FaceNet
##################
wnd_size = [220, 220]
x = tf.placeholder("float", shape=[None, wnd_size[0], wnd_size[1], 3])
y_ = tf.placeholder("float", shape=[None, 7])
x_image = x

######convolution layers
###layer 1
W_conv1 = weight_variable([7, 7, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d_reduce(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_3x3_reduce(h_conv1)  ## one layer 2x2 max pooling
h_norm1 = tf.contrib.layers.batch_norm(h_pool1)

####layer 2
W_conv2a = weight_variable([1, 1, 64, 64])
b_conv2a = bias_variable([64])
h_conv2a = tf.nn.relu(conv2d(h_norm1, W_conv2a) + b_conv2a)
W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_conv2a, W_conv2) + b_conv2)
h_norm2 = tf.contrib.layers.batch_norm(h_conv2)
h_pool2 = max_pool_3x3_reduce(h_norm2)  ## one layer 2x2 max pooling

###layer 3
W_conv3a = weight_variable([1, 1, 128, 128])
b_conv3a = bias_variable([128])
h_conv3a = tf.nn.relu(conv2d(h_pool2, W_conv3a) + b_conv3a)
W_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_conv3a, W_conv3) + b_conv3)
h_pool3 = max_pool_3x3_reduce(h_conv3)
h_norm3 = tf.contrib.layers.batch_norm(h_pool3)


###layer 4
W_conv4a = weight_variable([1, 1, 256, 256])
b_conv4a = bias_variable([256])
h_conv4a = tf.nn.relu(conv2d(h_norm3, W_conv4a) + b_conv4a)
W_conv4 = weight_variable([3, 3, 256, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_conv4a, W_conv4) + b_conv4)
h_norm4 = tf.contrib.layers.batch_norm(h_conv4)


###layer 5
W_conv5a = weight_variable([1, 1, 256, 256])
b_conv5a = bias_variable([256])
h_conv5a = tf.nn.relu(conv2d(h_norm4, W_conv5a) + b_conv5a)
W_conv5 = weight_variable([3, 3, 256, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_conv5a, W_conv5) + b_conv5)
h_norm5 = tf.contrib.layers.batch_norm(h_conv5)

####layer 6
W_conv6a = weight_variable([1, 1, 256, 256])
b_conv6a = bias_variable([256])
h_conv6a = tf.nn.relu(conv2d(h_norm5, W_conv6a) + b_conv6a)
W_conv6 = weight_variable([3, 3, 256, 256])
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv6a, W_conv6) + b_conv6)
h_pool6 = max_pool_3x3_reduce(h_conv6)
h_norm6 = tf.contrib.layers.batch_norm(h_pool6)

### fully connected layer
W_fc1 = weight_variable([7 * 7 * 256, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_norm6, [-1, 7 * 7 * 256])  # flat into 1 dimention
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # kill some neuron

# Readout Layer
W_fc2 = weight_variable([128, 7])
b_fc2 = bias_variable([7])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


###########training part
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

mse = tf.reduce_mean(tf.square(y_-y_conv))

## l2 norm loss
unregularized_loss = cross_entropy
#
l2_regularization_penalty = 1
l2_loss = l2_regularization_penalty *\
          (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2a) + tf.nn.l2_loss(W_conv2)
           + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv3a) + tf.nn.l2_loss(W_conv4) +
           tf.nn.l2_loss(W_conv4a) + tf.nn.l2_loss(b_conv5) + tf.nn.l2_loss(W_conv5a) +
           tf.nn.l2_loss(W_conv6) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

# l2_loss = l2_regularization_penalty *\
#           (tf.nn.l2_loss(W_conv1)  + tf.nn.l2_loss(W_conv2)
#            + tf.nn.l2_loss(W_conv3)  + tf.nn.l2_loss(W_conv4) +
#            + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

loss = tf.add(unregularized_loss, l2_loss)
############################

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#
saver.restore(sess, './onet_train.ckpt')
print("Model restored.")
step = 0
batch_i = 0
min_acc = 2.0
max_acc = 0.0
total_acc = 0
avg_acc = 0
stride = 50
file_i = 1
total_e = 0

while True:

    step += 1

    # if batch_i == 0:


    #s = list(zip(x_data, y_data))
    #random.shuffle(s)
    #x_data_batch, y_data_batch = zip(*s)

    if batch_i < 1:
        x_data = numpy.load(data_dir + 'img_data_id%d.npy' % file_i)
        y_data = numpy.load(data_dir + 'label_id%d.npy' % file_i)
        file_i += 1

    if x_data is None or y_data is None:
        print 'none type break'
        break

    x_data_batch = x_data[batch_i]
    y_data_batch = y_data[batch_i]
    #
    x_data_batch = x_data_batch.reshape(1, target_wnd_size[0], target_wnd_size[1], 3)
    y_data_batch = y_data_batch.reshape(1, 7)

    batch_i += 1

    if batch_i == 48:
        avg_acc = float(total_acc) / step
        avc_e = float(total_e) / step


        print 'file id is %d.' % (file_i - 1)
        print 'minimum accuracy is %f' % min_acc
        print 'maximum accuracy is %f' % max_acc
        print 'average accuracy is %f' % avg_acc
        print 'average error is %f' % avc_e

        print 'Save model'
        print

        batch_i = 0

        save_path = saver.save(sess, save_path= './onet_train.ckpt')


    if file_i > file_len:
        avg_acc = float(total_acc) / step
        avc_e = float(total_e) / step

        print 'minimum accuracy is %f' % min_acc
        print 'maximum accuracy is %f' % max_acc
        print 'average accuracy is %f' % avg_acc
        print 'average error is %f' % avc_e

        print 'Save model'
        print 'One Round'
        print

        save_path = saver.save(sess, save_path= './onet_train.ckpt')

        if avg_acc > 0.99:
            print 'Saving the last model and break'
            save_path = saver.save(sess, save_path='./onet_train.ckpt')
            break

        file_i = 1
        step = 0
        batch_i = 0
        min_acc = 2.0
        max_acc = 0.0
        total_acc = 0
        avg_acc = 0
        total_e = 0



    train_step.run({y_: y_data_batch, x: x_data_batch, keep_prob: 0.5}, sess)

    e = sess.run(mse, feed_dict={y_: y_data_batch, x: x_data_batch, keep_prob: 1.0})
    train_accuracy = accuracy.eval(feed_dict={y_: y_data_batch, x: x_data_batch, keep_prob: 1.0})



    if min_acc > train_accuracy:
        min_acc = train_accuracy

    if max_acc < train_accuracy:
        max_acc = train_accuracy

    total_acc += train_accuracy

    total_e += e


    # print 'batch number %d' % batch_i
    # print train_accuracy
    # print


    # if train_accuracy > 0.7:
    #     batch_i += 1

    # if train_accuracy > 0.99:
    #     print 'save model'
    #     save_path = saver.save(sess, save_path= './onet_train.ckpt')
    #     break



# ##############
# ## ONET Part Finishes
# ##############
