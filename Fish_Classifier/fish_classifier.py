from skimage import io, transform
import tensorflow as tf


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


def retrieve_prob_list(image_path):
    target_wnd_size = [220, 220]
    x = tf.placeholder("float", shape=[None, target_wnd_size[0], target_wnd_size[1], 3])
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
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # kill some neuron

    # Readout Layer
    W_fc2 = weight_variable([128, 7])
    b_fc2 = bias_variable([7])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'Fish_Classifier/onet_train.ckpt')
        image = io.imread(image_path)
        image_data = transform.resize(image, [target_wnd_size[0], target_wnd_size[1]])
        image_data = image_data.reshape([1, target_wnd_size[0], target_wnd_size[1], 3])
        validation_pred = y_conv.eval({x: image_data}, sess)

        return validation_pred
