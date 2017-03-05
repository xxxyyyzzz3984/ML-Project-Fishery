import random
import copy
from PIL import ImageFilter, Image
import pickle

import cv2
import numpy
from resizeimage import resizeimage
from os import listdir
from os.path import isfile, join
from skimage.morphology import watershed
from skimage.filters import sobel
from skimage.feature import hog
from skimage import data, color, exposure, filters, io, transform
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy import ndimage as ndi

from skimage.segmentation import random_walker

from os import listdir
from os.path import isfile, join

scan_wnd_size = [48, 48]


head_folder = '../../fish parts/head/'
head_pics = [f for f in listdir(head_folder)
               if isfile(join(head_folder, f))]

body_folder = '../../fish parts/body/'
body_pics = [f for f in listdir(body_folder)
               if isfile(join(body_folder, f))]

tail_folder = '../../fish parts/tail/'
tail_pics = [f for f in listdir(tail_folder)
               if isfile(join(tail_folder, f))]

wholefish_folder = '../../fish parts/whole_fish_horiz/'
fish_pics = [f for f in listdir(wholefish_folder)
               if isfile(join(wholefish_folder, f))]

array_save_dir = '../../array train dataset/'
training_fish_dir = '../../train dataset/'
no_fish_dir = 'NoF/'

# ######head part
# x_data_list = []
# for pic_name in fish_pics:
#     pic_filepath = wholefish_folder + pic_name
#     image = io.imread(pic_filepath)
#
#     image_data = transform.resize(image, scan_wnd_size)
#     image_data = image_data.reshape([scan_wnd_size[0] * scan_wnd_size[0], 3])
#     x_data_list .append(copy.copy(image_data))
#
# x_data = numpy.array(x_data_list)
# print x_data.shape
# numpy.save(array_save_dir + 'whole_fish_horiz_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)
# # ########################
#
# ######body part
# x_data_list = []
# for pic_name in body_pics:
#     pic_filepath = body_folder + pic_name
#     image = io.imread(pic_filepath)
#
#     image_data = transform.resize(image, scan_wnd_size)
#     image_data = image_data.reshape([scan_wnd_size[0] * scan_wnd_size[0], 3])
#     x_data_list .append(image_data)
#
# x_data = numpy.array(copy.copy(x_data_list))
# print x_data.shape
# numpy.save(array_save_dir + 'fish_body_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)
# ########################
#
# ######tail part
# x_data_list = []
# for pic_name in tail_pics:
#     pic_filepath = tail_folder + pic_name
#     image = io.imread(pic_filepath)
#
#     image_data = transform.resize(image, scan_wnd_size)
#     image_data = image_data.reshape([scan_wnd_size[0] * scan_wnd_size[0], 3])
#     x_data_list .append(copy.copy(image_data))
#
# x_data = numpy.array(x_data_list)
# print x_data.shape
# numpy.save(array_save_dir + 'fish_tail_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)
# # ########################

# no fish part
x_data_list = []
no_patch_eachpic = 10
nofish_pics = [f for f in listdir(training_fish_dir+no_fish_dir)
               if isfile(join(training_fish_dir+no_fish_dir, f))]

for nofish_pic_name in nofish_pics:
    image = io.imread(training_fish_dir+no_fish_dir+nofish_pic_name)


    for i in range(no_patch_eachpic):
        while True:
            try:
                y = random.randint(0, 720)
                x = random.randint(0, 1280)

                image_data = copy.copy(image[x:x+100, y:y+100, 0:3])

                image_data = transform.resize(image_data, numpy.array(scan_wnd_size))

                image_data = numpy.array(image_data, dtype=float)

                image_data = image_data.reshape(scan_wnd_size[0] * scan_wnd_size[1], 3)

                x_data_list.append(image_data)

                break
            except:
                pass
#
x_data = numpy.array(x_data_list)
print x_data.shape
numpy.save(array_save_dir+'nofish_imagedata_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)
