import numpy
from os import listdir
from os.path import isfile, join
from skimage import data, color, exposure, filters, io, transform
import cv2



fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
no_fish_dir = 'NoF/'


# get fish data part
x_data_list = []
scan_wnd_size = [48, 48]
count = 0
for fish_dir in fish_dirs:
    folder_path = '../../cropped train dataset/' + fish_dir + '/'
    fish_pics = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for fish_pic_name in fish_pics:
        image = io.imread(folder_path + fish_pic_name)
        w = image.shape[0]
        h = image.shape[1]

        if w > h:
            image = transform.rotate(image, 90)

        io.imsave('../../fish parts/whole_fish_horiz/%d.jpg' % count, image)
        count += 1


