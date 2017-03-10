import selectivesearch
import tensorflow as tf
import numpy
import copy
import os
from skimage import io, transform



'''Inspired by CVPR 16 paper, Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks'''

save_models_dir = '../saved models/'

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = numpy.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last],
                                                     numpy.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


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

def find_save_fish_onet(image_path, image_name, save_root_dir):
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

    keep_prob = tf.placeholder(tf.float32)

    W_fc2 = weight_variable([256, 2], name='wfc2_pnet')
    b_fc2 = bias_variable([2], name='bfc2_pnet')

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'Onet_Fish_Detector/reinforceonet_findfish_train.ckpt')
        print 'model restored'

        image = io.imread(image_path)

        print 'doing selective search...'
        _, regions = selectivesearch.selective_search \
            (image, scale=500, sigma=0.9, min_size=100)

        print 'finish selective search'

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue

            # distorted rects
            x_rect, y_rect, w, h = r['rect']

            try:
                if w / h > 1.2 or h / w > 1.2:
                    continue
            except ZeroDivisionError:
                continue

            # if r['size'] > image.shape[0] * image.shape[1] / 5:
            #     continue

            if h >= image.shape[0] / 2 or w >= image.shape[1] / 2:
                continue

            candidates.add(r['rect'])

        count = 1

        for x_rect, y_rect, w, h in candidates:

            image_data = copy.copy(image[y_rect:y_rect + h, x_rect:x_rect + w, 0:3])

            image_data = transform.resize(image_data, numpy.array(scan_wnd_size))
            image_data = numpy.array(image_data, dtype=float)
            image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1], 3)

            y_predict = y_conv.eval({x: image_data}, sess)

            if y_predict[0][0] > y_predict[0][1]:
                image_data = copy.copy(image[y_rect:y_rect + h, x_rect:x_rect + w, 0:3])
                mkdir_save_root_dir = save_root_dir.replace(' ', '\ ')
                os.system('mkdir ' + mkdir_save_root_dir + image_name + '/')
                io.imsave(save_root_dir + image_name + '/' + str(count) + '.jpg', image_data)
                count += 1


def pre_screen_is_fish(image_path):
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

    keep_prob = tf.placeholder(tf.float32)

    W_fc2 = weight_variable([256, 2], name='wfc2_pnet')
    b_fc2 = bias_variable([2], name='bfc2_pnet')

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'Onet_Fish_Detector/reinforceonet_findfish_train.ckpt')
        print 'model restored'

        image = io.imread(image_path)
        image_data = transform.resize(image, numpy.array(scan_wnd_size))
        image_data = numpy.array(image_data, dtype=float)
        image_data = image_data.reshape(1, scan_wnd_size[0] * scan_wnd_size[1], 3)

        y_predict = y_conv.eval({x: image_data}, sess)

        ## This is not a fish
        if y_predict[0][1] > y_predict[0][0]:
            return False

        else:
            return True
