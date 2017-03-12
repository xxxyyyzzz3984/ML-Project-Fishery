import numpy
from os import listdir
from os.path import isfile, join
from skimage import io, transform
import copy

fish_folders = ['../recropped train dataset/ALB/fish/', '../recropped train dataset/BET/fish/',
                '../recropped train dataset/DOL/fish/', '../recropped train dataset/LAG/fish/',
                '../recropped train dataset/OTHER/fish/', '../recropped train dataset/SHARK/fish',
                '../recropped train dataset/YFT/fish/']

nofish_folders = ['../recropped train dataset/ALB/nofish/', '../recropped train dataset/BET/nofish/',
                '../recropped train dataset/DOL/nofish/', '../recropped train dataset/LAG/nofish/',
                '../recropped train dataset/OTHER/nofish/', '../recropped train dataset/SHARK/nofish/',
                '../recropped train dataset/YFT/nofish/']

fish_npy_save_dir = '../array train dataset/recropped fish data/fish/'
nofish_npy_save_dir = '../array train dataset/recropped fish data/nofish/'

## fish part
# scan_wnd_size = [48, 48]
# image_list = []
# file_count = 1
# count = 1
# for fish_folder in fish_folders:
#     fish_imagenames = [f for f in listdir(fish_folder)
#                            if isfile(join(fish_folder, f))]
#
#     for fish_imagename in fish_imagenames:
#         image_path = fish_folder + fish_imagename
#         image = io.imread(image_path)
#         image_data = transform.resize(image, scan_wnd_size)
#         image_data = image_data.reshape([scan_wnd_size[0] * scan_wnd_size[1], 3])
#         image_data = numpy.array(image_data, dtype=float)
#         image_list.append(copy.copy(image_data))
#
#         if file_count % 50 == 0:
#             image_array = numpy.array(image_list, dtype=float)
#             numpy.save(fish_npy_save_dir + 'id%d_%dx%d.npy' %
#                        (count, scan_wnd_size[0], scan_wnd_size[1]), image_array)
#             image_list = []
#             count += 1
#             print image_array.shape
#
#         file_count += 1
#
# if len(image_list) > 0:
#     image_array = numpy.array(image_list, dtype=float)
#     numpy.save(fish_npy_save_dir + 'id%d_%dx%d.npy' %
#                (count + 1, scan_wnd_size[0], scan_wnd_size[1]), image_array)
#     image_list = []
#     print image_array

## no fish part
scan_wnd_size = [48, 48]
image_list = []
file_count = 1
count = 1
for nofish_folder in nofish_folders:
    fish_imagenames = [f for f in listdir(nofish_folder)
                           if isfile(join(nofish_folder, f))]

    for fish_imagename in fish_imagenames:
        image_path = nofish_folder + fish_imagename
        image = io.imread(image_path)
        image_data = transform.resize(image, scan_wnd_size)
        image_data = image_data.reshape([scan_wnd_size[0] * scan_wnd_size[1], 3])
        image_data = numpy.array(image_data, dtype=float)
        image_list.append(copy.copy(image_data))

        if file_count % 50 == 0:
            image_array = numpy.array(image_list, dtype=float)
            numpy.save(nofish_npy_save_dir + 'id%d_%dx%d.npy' %
                       (count, scan_wnd_size[0], scan_wnd_size[1]), image_array)
            image_list = []
            count += 1
            print image_array.shape

        file_count += 1

if len(image_list) > 0:
    image_array = numpy.array(image_list, dtype=float)
    numpy.save(nofish_npy_save_dir + 'id%d_%dx%d.npy' %
               (count + 1, scan_wnd_size[0], scan_wnd_size[1]), image_array)
    image_list = []
    print image_array