
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
import collections
get_ipython().magic('matplotlib inline')


# In[2]:

validation = pickle.load( open('/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/validation.p', "rb" ) )


# In[3]:

# validation set
X_val = np.asarray(validation[1])


# In[4]:

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


# In[5]:

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
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc2 = weight_variable([128, 7])
b_fc2 = bias_variable([7])
y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# training
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[60]:

save_models_dir = '/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/save/11/'


# In[61]:

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_models_dir + 'fish_classification.ckpt')    
    validation_pred = y_conv.eval({x: X_val}, sess) 


# In[62]:

folders = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
Y_val = np.asarray(validation[2])
X_id_val = validation[0]
label_val = validation[3]


# In[63]:

predict_result = np.array(validation_pred)
score = sum(-sum(Y_val*np.log(predict_result))/len(predict_result))

correct_prediction = np.equal(np.argmax(predict_result,axis = 1), np.argmax(Y_val,axis = 1)) 
correct_count = collections.Counter(correct_prediction)
accuracy = correct_count[True]/len(correct_prediction)

print('The score is:', score,'The accuracy is:', accuracy)


# In[64]:

df = pd.DataFrame(validation_pred,index=X_id_val, columns = folders )
df['ground_truth'] = label_val
df['T/F'] = correct_prediction


# In[66]:

df.sort_index(inplace=True)


# In[67]:

df.head(20)


# In[68]:

df.to_csv('/Users/Shuyuan/Desktop/fish_validation.csv', sep =',')


# In[ ]:



