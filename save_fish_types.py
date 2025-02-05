import random

import numpy
import copy
from skimage import io, transform, color, feature
import os
from os.path import isfile, join

root_dir = '../train dataset/'

croppeed_fish_folders = [root_dir + 'ALB/', root_dir + 'BET/',
                         root_dir + 'DOL/', root_dir+'LAG/', root_dir + 'NoF/',
                         root_dir + 'OTHER/', root_dir + 'SHARK/',
                         root_dir + 'YFT/']

all_fish_image_paths = []


wnd_size = [220, 220]

save_dir = '../array train dataset/fish types/whole_%dx%d/' % (wnd_size[0], wnd_size[1])


for croppeed_fish_folder in croppeed_fish_folders:
    train_fish_images = [f for f in os.listdir(croppeed_fish_folder)
                         if isfile(join(croppeed_fish_folder, f))]

    #img_list = []
    #for train_fish_image in train_fish_images:
    #    image_path = croppeed_fish_folder + train_fish_image
    #    image_data = io.imread(image_path)
    #    image_data = transform.resize(image_data, wnd_size)
    #    image_data = image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    #    image_data = numpy.array(image_data, dtype=float)
    #    img_list.append(image_data)

    # img_array = numpy.array(img_list)
    # print img_array.shape
    # numpy.save(save_dir + croppeed_fish_folder.replace('../cropped train dataset/', '').replace('/', ''),
    #            img_array)


    image_paths = []
    for train_fish_image in train_fish_images:
        image_path = croppeed_fish_folder + train_fish_image
        image_paths.append(image_path)

    all_fish_image_paths.append(copy.copy(image_paths))

ALB_len = len(all_fish_image_paths[0])
BET_len = len(all_fish_image_paths[1])
DOL_len = len(all_fish_image_paths[2])
LAG_len = len(all_fish_image_paths[3])
NOF_len = len(all_fish_image_paths[4])
OTHER_len = len(all_fish_image_paths[5])
SHARK_len = len(all_fish_image_paths[6])
YFT_len = len(all_fish_image_paths[7])

step1, step2, step3, step4, step5, step6, step7, step8 = [0, 0, 0, 0, 0, 0, 0, 0]

img_list = []
label_list = []
file_count = 1
file_size = 48
for i in range(max(ALB_len, BET_len, DOL_len, LAG_len, OTHER_len, SHARK_len, YFT_len)):
    ALB_image_data = io.imread(all_fish_image_paths[0][step1])
    BET_image_data = io.imread(all_fish_image_paths[1][step2])
    DOL_image_data = io.imread(all_fish_image_paths[2][step3])
    LAG_image_data = io.imread(all_fish_image_paths[3][step4])
    NOF_image_data = io.imread(all_fish_image_paths[4][step5])
    OTHER_image_data = io.imread(all_fish_image_paths[5][step6])
    SHARK_image_data = io.imread(all_fish_image_paths[6][step7])
    YFT_image_data = io.imread(all_fish_image_paths[7][step8])


    #print all_fish_image_paths[0][step1]

    ALB_image_data = transform.resize(ALB_image_data, wnd_size)
    BET_image_data = transform.resize(BET_image_data, wnd_size)
    DOL_image_data = transform.resize(DOL_image_data, wnd_size)
    LAG_image_data = transform.resize(LAG_image_data, wnd_size)
    NOF_image_data = transform.resize(NOF_image_data, wnd_size)
    OTHER_image_data = transform.resize(OTHER_image_data, wnd_size)
    SHARK_image_data = transform.resize(SHARK_image_data, wnd_size)
    YFT_image_data = transform.resize(YFT_image_data, wnd_size)


    if ALB_image_data.shape != (220, 220, 3):
        print all_fish_image_paths[0][step1]
        print step1
        print
        print ALB_image_data.shape

    if BET_image_data.shape != (220, 220, 3):
        print all_fish_image_paths[1][step2]
        print BET_image_data.shape

    if DOL_image_data.shape != (220, 220, 3):
        print all_fish_image_paths[2][step3]
        print DOL_image_data.shape

    if LAG_image_data.shape != (220, 220, 3):
        print all_fish_image_paths[3][step4]
        print LAG_image_data.shape

    if SHARK_image_data.shape != (220, 220, 3):
        print all_fish_image_paths[5][step6]
        print SHARK_image_data.shape

    if YFT_image_data.shape != (220, 220, 3):
        print all_fish_image_paths[6][step7]
        print YFT_image_data.shape


    # ALB_image_data = ALB_image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    # BET_image_data = BET_image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    # DOL_image_data = DOL_image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    # LAG_image_data = LAG_image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    # OTHER_image_data = OTHER_image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    # SHARK_image_data = SHARK_image_data.reshape(wnd_size[0] * wnd_size[1], 3)
    # YFT_image_data = YFT_image_data.reshape(wnd_size[0] * wnd_size[1], 3)

    # ALB_image_data = numpy.array(ALB_image_data, dtype=float)
    # BET_image_data = numpy.array(BET_image_data, dtype=float)
    # DOL_image_data = numpy.array(DOL_image_data, dtype=float)
    # LAG_image_data = numpy.array(LAG_image_data, dtype=float)
    # OTHER_image_data = numpy.array(OTHER_image_data, dtype=float)
    # SHARK_image_data = numpy.array(SHARK_image_data, dtype=float)
    # YFT_image_data = numpy.array(YFT_image_data, dtype=float)

    img_list.append(copy.copy(ALB_image_data))
    img_list.append(copy.copy(BET_image_data))
    img_list.append(copy.copy(DOL_image_data))
    img_list.append(copy.copy(LAG_image_data))
    img_list.append(copy.copy(NOF_image_data))
    img_list.append(copy.copy(OTHER_image_data))
    img_list.append(copy.copy(SHARK_image_data))
    img_list.append(copy.copy(YFT_image_data))

    label_list.append(numpy.array([2, 0, 0, 0, 0, 0, 0, 0]))
    label_list.append(numpy.array([0, 2, 0, 0, 0, 0, 0, 0]))
    label_list.append(numpy.array([0, 0, 2, 0, 0, 0, 0, 0]))
    label_list.append(numpy.array([0, 0, 0, 2, 0, 0, 0, 0]))
    label_list.append(numpy.array([0, 0, 0, 0, 2, 0, 0, 0]))
    label_list.append(numpy.array([0, 0, 0, 0, 0, 2, 0, 0]))
    label_list.append(numpy.array([0, 0, 0, 0, 0, 0, 2, 0]))
    label_list.append(numpy.array([0, 0, 0, 0, 0, 0, 0, 2]))


    step1 += 1
    step2 += 1
    step3 += 1
    step4 += 1
    step5 += 1
    step6 += 1
    step7 += 1
    step8 += 1

    if step1 >= ALB_len:
        print 'herer'
        print step1
        step1 = 0

    if step2 >= BET_len:
        step2 = 0

    if step3 >= DOL_len:
        step3 = 0

    if step4 >= LAG_len:
        step4 = 0

    if step5 >= NOF_len:
        step5 = 0

    if step6 >= OTHER_len:
        step6 = 0

    if step7 >= SHARK_len:
        step7 = 0

    if step8 >= YFT_len:
        step8 = 0

    if len(img_list) >= file_size:

         img_array = numpy.array(img_list)
         label_array = numpy.array(label_list)
         s = list(zip(img_array, label_array))
         random.shuffle(s)
         img_array_shuffle, label_array_shuffle = zip(*s)


         print img_array.shape
         print label_array.shape

         numpy.save(save_dir + 'img_data_id%d' % file_count, img_array_shuffle)
         numpy.save(save_dir + 'label_id%d' % file_count, label_array_shuffle)

         img_list = []
         label_list = []

         file_count += 1
