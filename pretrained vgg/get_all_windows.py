
# coding: utf-8

# In[1]:

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy.misc
import os
get_ipython().magic('matplotlib inline')


# In[2]:

def get_im(path):
    image = Image.open(path)
    arr = np.asarray(image)
    return arr


# In[3]:

def get_windows(image, sz, stride):
    cropped = {}
    i = 0
    j = 0
    while j*stride + sz <= image.shape[1]:
        while i*stride + sz <= image.shape[0]:
            img = image[i*stride:(i*stride + sz),j*stride:(j*stride + sz)]
            cropped[i,j] = img
            i += 1
            n_l = i
        i = 0
        j += 1
        n_w = j
    j = 0
    while j*stride + sz <= image.shape[1]:
        img = image[image.shape[0]-sz:image.shape[0],j*stride:(j*stride + sz)]
        cropped[n_l,j] = img
        j += 1
    i = 0
    while i*stride + sz <= image.shape[0]:
        img = image[i*stride:(i*stride + sz),image.shape[1]-sz:image.shape[1]]
        cropped[i,n_w] = img
        i += 1
    cropped[n_l,n_w] = image[image.shape[0]-sz:image.shape[0],image.shape[1]-sz:image.shape[1]]
    
    return cropped, [n_l, n_w]


# In[4]:

def save_coropped(file_name,path_root,save_path,sz,stride):
    
    path = os.path.join(path_root, file_name + '.jpg')
    image = get_im(path)
    img, size = get_windows(image,size,stride)
    for key, value in img.items():
        path = os.path.join(save_path, file_name+'_'+str(key[0])+'-'+str(key[1])+'.jpg')
        scipy.misc.imsave(path, value)
    print(size)


# In[5]:

#path_root = '../NCFM_data/test_stg1/'
#save_path = './test/04/'


# In[6]:

#file_name = 'img_00007'
#sz = 500
#stride = 100


# In[7]:

#save_coropped(file_name,path_root,save_path,sz,stride)


# In[ ]:



