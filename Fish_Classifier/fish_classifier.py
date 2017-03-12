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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def retrieve_prob_list(image_path):
    x = tf.placeholder("float", shape=[None, 48 * 48, 3])
    y_ = tf.placeholder("float", shape=[None, 7])
    x_image = tf.reshape(x, [-1, 48, 48, 3])

    # convolutional layers
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_3x3(h_conv1)  ## one layer 3x3 max pooling

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2)  ## one layer 2x2 max pooling

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)  ## one layer 2x2 max pooling

    W_conv4 = weight_variable([3, 3, 64, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

    # fully connected layer
    W_fc1 = weight_variable([3 * 3 * 128, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(h_conv4, [-1, 3 * 3 * 128])  # flat into 1 dimention
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # kill some neuron

    # Readout Layer
    W_fc2 = weight_variable([256, 7])
    b_fc2 = bias_variable([7])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'Fish_Classifier/onet_train.ckpt')
        image = io.imread(image_path)
        image_data = transform.resize(image, [48, 48])
        image_data = image_data.reshape([1, 48, 48, 3])

        validation_pred = y_conv.eval({x: image_data}, sess)

        return validation_pred