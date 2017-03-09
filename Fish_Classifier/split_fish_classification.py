
# coding: utf-8

# In[1]:

import os
import numpy as np
import shutil
from PIL import Image
import PIL
import glob
import pickle
import pandas as pd


# In[2]:

def get_im(path):
    image = Image.open(path)
    image =image.resize((48,48), PIL.Image.ANTIALIAS)
    arr = np.asarray(image)/255
    return arr


# In[3]:

def data_split(source_root, split_proportion):
    np.random.seed(2017)
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
    train = []
    validation = []
    
    for fish in folders:
        total_images = os.listdir(os.path.join(source_root, fish))
        L = len(total_images)
        i = 1
        while L < 1000:
            total_images = total_images * i
            i += 1
            L = len(total_images)
        total_images = np.random.choice(total_images, 1000)
        
        nbr_train = int(len(total_images) * split_proportion)
        np.random.shuffle(total_images)
        train_images = total_images[:nbr_train]
        val_images = total_images[nbr_train:]
        label_index = folders.index(fish)
              
        for fl in train_images:
            path = os.path.join(source_root, fish, fl)
            img = get_im(path)
            train.append([fl, img, label_index]) # X_id_train, X_train, X_label_index
 
        for fl in val_images:
            path = os.path.join(source_root, fish, fl)
            img = get_im(path)
            validation.append([fl, img, label_index])
           
    np.random.shuffle(train)
    np.random.shuffle(validation)
    
    X_id_train = []
    X_trian = []
    label_train = []
    
    for i in range(len(train)):
        X_id_train.append(train[i][0])
        X_trian.append(train[i][1])
        label_train.append(train[i][2])
    
    Y_train = pd.get_dummies(label_train)
    train_set = [X_id_train, X_trian, Y_train, label_train]
    
    X_id_val = []
    X_val = []
    label_val = []
    
    for i in range(len(validation)):
        X_id_val.append(validation[i][0])
        X_val.append(validation[i][1])
        label_val.append(validation[i][2])
    
    Y_val = pd.get_dummies(label_val)
    validation_set = [X_id_val, X_val, Y_val, label_val]
    
    return train_set, validation_set


# In[4]:

train, validation = data_split('/Users/Shuyuan/Desktop/NCFM_data/cropped_train', 0.95)


# In[5]:

pickle.dump(train, open('/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/train.p', 'wb'))


# In[6]:

pickle.dump(validation, open('/Users/Shuyuan/Desktop/NCFM/saved_model/second_one/validation.p', 'wb'))


# In[ ]:



