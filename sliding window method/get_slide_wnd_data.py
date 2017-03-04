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

training_parsing_json_dir = '../dataset parsing/'
cropped_train_dir = '../cropped train dataset/'
training_fish_dir = '../../train dataset/'
array_save_dir = '../../array train dataset/'

fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
no_fish_dir = 'NoF/'


# get fish data part
# x_data_list = []
# y_data_list = []
# scan_wnd_size = [100, 100]
# search_stride = scan_wnd_size[0]
# iteration = 0
# for fish_dir in fish_dirs:
#     folder_path = '../../cropped train dataset/' + fish_dir + '/'
#     fish_pics = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
#     for fish_pic_name in fish_pics:
#         image = io.imread(folder_path + fish_pic_name)
#
#         i_total = image.shape[0] / search_stride
#         j_total = image.shape[1] / search_stride
#
#         for i in range(i_total):
#             for j in range(j_total):
#                 image_data = copy.copy(image[i * search_stride:i * search_stride + scan_wnd_size[0],
#                                        j * search_stride:j * search_stride + scan_wnd_size[1], 0:3])
#
#                 if image_data.shape != (scan_wnd_size[0], scan_wnd_size[0], 3):
#                     continue
#
#                 image_data = numpy.array(image_data, dtype=float)
#
#                 image_data = image_data.reshape(scan_wnd_size[0] * scan_wnd_size[1], 3)
#
#                 # print image_data.shape
#
#                 x_data_list.append(copy.copy(image_data))
#                 y_data_list.append([1, 0])
#
#                 iteration += 1
#
#                 if iteration % 50 == 0:
#                     x_data = numpy.array(x_data_list)
#                     numpy.save(array_save_dir + 'big_pic_blocks/it%d_image_%dx%d.npy' % (iteration / 50,
#                                                                                          scan_wnd_size[0],
#                                                                                          scan_wnd_size[1]), x_data)
#
#                     print x_data.shape
#                     y_data = numpy.array(y_data_list)
#                     numpy.save(array_save_dir + 'big_pic_blocks/it%d_labels_%dx%d.npy' % (iteration / 50,
#                                                                                           scan_wnd_size[0],
#                                                                                           scan_wnd_size[1]), y_data)
#
#                     print y_data.shape
#
#                     x_data_list = []
#                     y_data_list = []
#
# x_data = numpy.array(x_data_list)
# numpy.save(array_save_dir + 'big_pic_blocks/it%d_image_%dx%d.npy' % ((iteration/50)+1,
#                     scan_wnd_size[0], scan_wnd_size[1]), x_data)
#
# print x_data.shape
# y_data = numpy.array(y_data_list)
# numpy.save(array_save_dir + 'big_pic_blocks/it%d_labels_%dx%d.npy' % ((iteration/50)+1,
#                         scan_wnd_size[0], scan_wnd_size[1]), y_data)
#
# print y_data.shape


# x_data = numpy.array(x_data_list)
# print x_data.shape
# numpy.save(array_save_dir+'fish_imagedata_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)
#
#
# with open(array_save_dir + 'fish_hog_%dx%d.pkl'%(scan_wnd_size[0], scan_wnd_size[1]), 'wb') as fp:
#     pickle.dump(x_data_list, fp)

# no fish part
iteration = 0
x_data_list = []
scan_wnd_size = [100, 100]
no_patch_eachpic = 25
nofish_pics = [f for f in listdir(training_fish_dir+no_fish_dir)
               if isfile(join(training_fish_dir+no_fish_dir, f))]

for nofish_pic_name in nofish_pics:
    image = io.imread(training_fish_dir+no_fish_dir+nofish_pic_name)

    # edges_data = canny(image_grey)
    # image_grey = filters.gaussian(edges_data, 2)

    for i in range(no_patch_eachpic):
        while True:
            try:
                y = random.randint(0, 720)
                x = random.randint(0, 1280)

                image_data = copy.copy(image[x:x+scan_wnd_size[0], y:y+scan_wnd_size[1], 0:3])

                image_data = numpy.array(image_data, dtype=float)

                image_data = image_data.reshape(scan_wnd_size[0] * scan_wnd_size[1], 3)

                x_data_list.append(copy.copy(image_data))

                break
            except:
                pass

    iteration += 1

    if iteration % 2 == 0:
        x_data = numpy.array(x_data_list)
        numpy.save(array_save_dir + 'big_pic_blocks/it%d_image_%dx%d.npy' % ((iteration / 2),
                        scan_wnd_size[0], scan_wnd_size[1]),
                   x_data)

        print x_data.shape

        x_data_list = []


x_data = numpy.array(x_data_list)
print x_data.shape
numpy.save(array_save_dir+'nofish_imagedata_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)



