
# coding: utf-8

# In[1]:

import os
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import tensorflow as tf
import shutil
from scipy.signal import convolve2d
from PIL import Image
import PIL
import glob
import math
import pickle


# In[2]:

train = pickle.load( open('/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/train.p', "rb" ) )


# In[3]:

validation = pickle.load( open('/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/validation.p', "rb" ) )


# In[4]:

X = np.asarray(train[1])
Y = np.asarray(train[2])
X_id = train[0]
label = train[3]
# validation set
X_val = np.asarray(validation[1])
Y_val = np.asarray(validation[2])
X_id_val = validation[0]
label_val = validation[3]


# In[5]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='SAME')


# In[6]:

#sess = tf.InteractiveSession()
x  = tf.placeholder("float", shape=[None, 48, 48, 3])
y_ = tf.placeholder("float", shape=[None, 7])
x_image = tf.reshape(x, [-1,48,48,3])

# convolutional layers
W_conv1 = weight_variable([5, 5, 3, 52])
b_conv1 = bias_variable([52])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_3x3(h_conv1)  ## one layer 3x3 max pooling

W_conv2 = weight_variable([5, 5, 52, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  ## one layer 2x2 max pooling

W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)  ## one layer 2x2 max pooling

W_conv4 = weight_variable([3, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)


# fully connected layer
W_fc1 = weight_variable([4 * 4 * 128, 128])  
b_fc1 = bias_variable([128])
 
h_pool2_flat = tf.reshape(h_conv4, [-1, 4 * 4 * 128])  # flat into 1 dimention
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # kill some neuron

# Readout Layer
W_fc2 = weight_variable([128, 7])
b_fc2 = bias_variable([7])

y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


# In[7]:

# training
#cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluating the Model
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[27]:

save_models_dir = '/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/save/'
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, '/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/save/10/fish_classification.ckpt')
batch_sz = 50
i  = 0
j = 0
print('j = ',j)
while True:
    if j > 5:       
        print('The loop is finished...')
        break
        
    if i * batch_sz > X.shape[0]:
        i = 0
        j += 1
        print('j = ',j)
                
    xs = X[i*batch_sz:(i*batch_sz + batch_sz),]
    ys = Y[i*batch_sz:(i*batch_sz + batch_sz),]
    
    if i%10 == 0:
        train_entropy = sess.run( cross_entropy, feed_dict={x:xs, y_: ys, keep_prob: 1.0})
        print("step %d, cross entropy %g"%(i, train_entropy))
        
    sess.run(train_step,feed_dict={x: xs, y_: ys, keep_prob: 0.5})
    
    i += 1
    if j%2 == 0 and i == 131:
        basic = str(j + 11)
        path = os.path.join(save_models_dir, basic)
        os.mkdir(path)
        PATH = os.path.join(path, 'fish_classification.ckpt')
        save_model = saver.save(sess, save_path = PATH)
        print('result saved')
        print('This iteration end ....')


# In[ ]:



