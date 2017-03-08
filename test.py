import json
from PIL import Image
import random
import copy
from PIL import ImageFilter, Image
import pickle
import matplotlib.patches as mpatches
import selectivesearch




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
training_fish_dir = '../train dataset/'


array_save_dir = '../array train dataset/'

fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
no_fish_dir = 'NoF/'

js_filenames = ['ALB.json', 'BET.json', 'DOL.json', 'LAG.json', 'OTHER.json', 'SHARK.json', 'YFT.json']
# js_filenames = ['ALB.json']

scan_wnd_size = [48, 48]
x_data_list = []
y_data_list = []

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
        image = io.imread(training_fish_dir + each_fish_data_js['filename'])
        img_lbl, regions = selectivesearch.selective_search \
            (image, scale=500, sigma=0.9, min_size=10)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue

            # excluding regions smaller than 2000 pixels
            if r['size'] < 6000:
                continue

            # distorted rects
            x_rect, y_rect, w, h = r['rect']

            # if r['size'] > image.shape[0] * image.shape[1] / 5:
            #     continue
            #
            if h >= image.shape[0] / 2 or w >= image.shape[1] / 2:
                continue

            candidates.add(r['rect'])


        # image_data = transform.resize(image_data, numpy.array(scan_wnd_size))
        # image_data = numpy.array(image_data, dtype=float)
        # image_data = image_data.reshape(scan_wnd_size[0] * scan_wnd_size[1], 3)
        plt.imshow(image)
        ax = plt.gca()

        for fish_info_js in fish_info_list:
            x = fish_info_js['x']
            y = fish_info_js['y']
            width = fish_info_js['width']
            height = fish_info_js['height']

            for x_ss, y_ss, w_ss, h_ss in candidates:
                rect = mpatches.Rectangle((x_ss, y_ss), w_ss, h_ss,
                                          fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)
                if x_ss <= x and y_ss <= y and x_ss+w_ss >= x+width-50 and y_ss+h_ss >= y+height-50:
                    rect = mpatches.Rectangle((x_ss, y_ss), w_ss, h_ss,
                                              fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
        plt.show()





        # y_data_list.append(numpy.array([min_x, min_y, max_w-min_x, max_h-min_y]))
        #
        # x_data_list.append(image_data)

# x_data = numpy.array(x_data_list)
# numpy.save('../array train dataset/fish_wholeimg_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), x_data)
#
# print x_data.shape
# y_data = numpy.array(y_data_list)
# numpy.save('../array train dataset/fishlabels_wholeimg_%dx%d.npy' % (scan_wnd_size[0], scan_wnd_size[1]), y_data)
#
# print y_data.shape

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