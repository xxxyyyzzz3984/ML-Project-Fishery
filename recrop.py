import os
from os import listdir
from os.path import isfile, join
import copy
from pandas import json
from collections import namedtuple
from skimage import io
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx*dy

def check_rect_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    r1 = Rectangle(x1, y1, x1+w1, y1+h1)
    r2 = Rectangle(x2, y2, x2+w2, y2+h2)

    if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
        return False
    else:
        common_area = area(r1, r2)
        max_intersect_ratio = max(float(common_area)/float(w1*h1), float(common_area)/float(w2*h2))
        if max_intersect_ratio > 0.2:
            return True
        else:
            return False

fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

js_filenames = ['ALB.json', 'BET.json', 'DOL.json', 'LAG.json', 'OTHER.json', 'SHARK.json', 'YFT.json']

training_parsing_json_dir = '../dataset parsing/'

fish_image_dict = dict()
for js_file in js_filenames:
    js_f = open(training_parsing_json_dir + js_file, 'r')

    lines = ''
    for line in js_f:
        lines += line.replace('\n', '')

    fish_js_data = json.loads(lines)

    for i in range(len(fish_js_data)):
        each_fish_data_js = fish_js_data[i]

        fish_info_list = each_fish_data_js['annotations']
        fish_image_dict[each_fish_data_js['filename']] = []
        for fish_info in fish_info_list:
            fish_image_dict[each_fish_data_js['filename']].\
                append([fish_info['x'], fish_info['y'], fish_info['width'], fish_info['height']])

all_fish_folders = [x[0] for x in os.walk('../train dataset/')]
all_fish_folders = all_fish_folders[1:len(all_fish_folders)]

count = 1

for fish_folder in all_fish_folders:
    if 'NoF' in fish_folder:
        continue

    fish_folder = fish_folder + '/'
    os.system('mkdir ' + '../recropped\ train\ dataset/' + fish_folder.replace('../train dataset/', ''))
    os.system('mkdir ' + '../recropped\ train\ dataset/' + fish_folder.replace('../train dataset/', '') + 'fish/')
    os.system('mkdir ' + '../recropped\ train\ dataset/' + fish_folder.replace('../train dataset/', '') + 'nofish/')


    save_dir = '../recropped train dataset/' + fish_folder.replace('../train dataset/', '')

    all_fish_imagenames = [f for f in listdir(fish_folder)
                     if isfile(join(fish_folder, f))]

    for fish_imagename in all_fish_imagenames:
        fish_image_path = fish_folder + fish_imagename
        fish_image = io.imread(fish_image_path)

        print 'processing ' + fish_image_path

        search_stride = 100
        i_w = 400
        j_w = 400

        i_total = fish_image.shape[0] / search_stride
        j_total = fish_image.shape[1] / search_stride

        for i in range(i_total):
            for j in range(j_total):
                image_data = copy.copy(fish_image[i * search_stride:i * search_stride + i_w,
                       j * search_stride:j * search_stride + j_w, 0:3])

                if image_data.shape != (400, 400, 3):
                    continue

                has_fish = False

                for x, y, w, h in fish_image_dict[fish_folder.
                        replace('../train dataset/', '') + fish_imagename]:

                    if check_rect_overlap(x, y, w, h,
                           j * search_stride, i * search_stride, j_w, i_w):
                        io.imsave(save_dir + 'fish/' + str(count) + '.jpg', image_data)

                        has_fish = True
                        break

                if not has_fish:
                    io.imsave(save_dir + 'nofish/' + str(count) + '.jpg', image_data)


                count += 1