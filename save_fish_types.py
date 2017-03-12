import numpy

from skimage import io, transform
import os
from os.path import isfile, join

croppeed_fish_folders = ['../cropped train dataset/ALB/', '../cropped train dataset/BET/',
                         '../cropped train dataset/DOL/', '../cropped train dataset/LAG/',
                         '../cropped train dataset/OTHER/', '../cropped train dataset/SHARK/',
                         '../cropped train dataset/YFT/']

wnd_size = [220, 220]

for croppeed_fish_folder in croppeed_fish_folders:
    train_fish_images = [f for f in os.listdir(croppeed_fish_folder)
                         if isfile(join(croppeed_fish_folder, f))]

    img_list = []
    for train_fish_image in train_fish_images:
        image_path = croppeed_fish_folder + train_fish_image
        image_data = io.imread(image_path)
        image_data = transform.resize(image_data, wnd_size)
        image_data = image_data.reshape(wnd_size[0] * wnd_size[1], 3)
        image_data = numpy.array(image_data, dtype=float)
        img_list.append(image_data)

    img_array = numpy.array(img_list)
    print img_array.shape
    numpy.save(croppeed_fish_folder.replace('../cropped train dataset/', '').replace('/', ''),
               img_array)