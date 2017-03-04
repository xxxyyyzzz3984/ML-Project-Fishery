import json
from PIL import Image
import random
import copy
from PIL import ImageFilter, Image
import pickle
import matplotlib.patches as mpatches


import cv2



import numpy
from resizeimage import resizeimage
from os import listdir
from os.path import isfile, join
from skimage.morphology import watershed
from skimage.filters import sobel
from skimage.feature import hog
from skimage import data, color, exposure, filters, io, transform, segmentation
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy import ndimage as ndi

from skimage.segmentation import random_walker

training_parsing_json_dir = '../dataset parsing/'
cropped_train_dir = '../cropped train dataset/'
training_fish_dir = '../train dataset/'


array_save_dir = '../array train dataset/'

fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
no_fish_dir = 'NoF/'

js_filenames = ['ALB.json', 'BET.json', 'DOL.json', 'LAG.json', 'OTHER.json', 'SHARK.json', 'YFT.json']
# js_filenames = ['ALB.json']

scan_wnd_size = [48, 48]
x_data_list = []
y_data_list = []

iteration = 0

for js_file in js_filenames:
    js_f = open(training_parsing_json_dir + js_file, 'r')

    lines = ''
    for line in js_f:
        lines += line.replace('\n', '')

    fish_js_data = json.loads(lines)

    for i in range(len(fish_js_data)):
        each_fish_data_js = fish_js_data[i]

        fish_info_list = each_fish_data_js['annotations']
        # im = Image.open(training_fish_dir + each_fish_data_js['filename'])
        image_data = io.imread(training_fish_dir + each_fish_data_js['filename'])
        # image_data = color.rgb2grey(image_data)

        # image_data = filters.canny(image_data)
        # image_data = filters.gaussian(image_data, 4)

        # image_data = filters.prewitt(image_data)


        # cv2.imshow('test', image_data)
        # cv2.waitKey()
        # image_data = canny(image_data)
        ori_size = copy.copy(image_data.shape)
        image_data = transform.resize(image_data, numpy.array(scan_wnd_size))
        image_data = numpy.array(image_data, dtype=float)
        image_data = image_data.reshape(scan_wnd_size[0] * scan_wnd_size[1], 3)
        # min_x = 2000
        # min_y = 2000
        # max_h = 0
        # max_w = 0

        for fish_info_js in fish_info_list:
            x = fish_info_js['x']
            y = fish_info_js['y']
            width = fish_info_js['width']
            height = fish_info_js['height']

            # if min_x > x:
            #     min_x = x
            #
            # if min_y > y:
            #     min_y = y
            #
            # if max_h < y + height:
            #     max_h = y + height
            #
            # if max_w < x + width:
            #     max_w = x + width


        # plt.imshow(io.imread(training_fish_dir + each_fish_data_js['filename']))
        # ax = plt.gca()
        # rect = mpatches.Rectangle((min_x, min_y),
        #                           max_w-min_x, max_h-min_y,
        #                           fill=False, edgecolor='red', linewidth=2)
        #
        # ax.add_patch(rect)
        # plt.show()

        # w = max_w-min_x
        # h = max_h-min_y

        x = float((x * scan_wnd_size[1]) / ori_size[1])
        y = float((y * scan_wnd_size[0]) / ori_size[0])

        w = float((width * scan_wnd_size[1]) / ori_size[1])
        h = float((height * scan_wnd_size[0]) / ori_size[0])

        y_data_list.append(numpy.array([x, y, w, h]))
        x_data_list.append(copy.copy(image_data))


        iteration += 1

        if iteration % 50 == 0:
            x_data = numpy.array(x_data_list)
            numpy.save(array_save_dir + 'big_pic_blocks/it%d_image_%dx%d.npy' % (iteration/50,
                                scan_wnd_size[0], scan_wnd_size[1]), x_data)

            print x_data.shape
            y_data = numpy.array(y_data_list)
            numpy.save(array_save_dir + 'big_pic_blocks/it%d_labels_%dx%d.npy' % (iteration/50,
                                    scan_wnd_size[0], scan_wnd_size[1]), y_data)

            print y_data.shape

            x_data_list = []
            y_data_list = []



x_data = numpy.array(x_data_list)
numpy.save(array_save_dir + 'big_pic_blocks/it%d_image_%dx%d.npy' % ((iteration/50)+1,
                    scan_wnd_size[0], scan_wnd_size[1]), x_data)

print x_data.shape
y_data = numpy.array(y_data_list)
numpy.save(array_save_dir + 'big_pic_blocks/it%d_labels_%dx%d.npy' % ((iteration/50)+1,
                        scan_wnd_size[0], scan_wnd_size[1]), y_data)

print y_data.shape

# x_data_list = []
# scan_wnd_size = [48, 48]
# y_data_list = []
# no_patch_eachpic = 10
# nofish_pics = [f for f in listdir(training_fish_dir+no_fish_dir)
#                if isfile(join(training_fish_dir+no_fish_dir, f))]
#
# for nofish_pic_name in nofish_pics:
#     # image_data = io.imread(training_fish_dir+no_fish_dir+nofish_pic_name)
#     # image_data = transform.resize(image_data, numpy.array(scan_wnd_size))
#     # image_data = numpy.array(image_data, dtype=float)
#     # image_data = image_data.reshape(scan_wnd_size[0] * scan_wnd_size[1], 3)
#
#     for i in range(10):
#         y_data_list.append(numpy.array([0, 1]))
#         # x_data_list.append(image_data)
#
# y_data = numpy.array(y_data_list)
# print y_data.shape
#
# # x_data = numpy.array(x_data_list)
# # print x_data.shape
# numpy.save('../array train dataset/nofishlabels_wholeimg_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), y_data)