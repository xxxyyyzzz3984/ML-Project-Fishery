import random
import copy
from PIL import ImageFilter, Image

import numpy
from resizeimage import resizeimage
from os import listdir
from os.path import isfile, join

training_parsing_json_dir = '../dataset parsing/'
cropped_train_dir = '../cropped train dataset/'
training_fish_dir = '../train dataset/'
array_save_dir = '../array train dataset/'

fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
no_fish_dir = 'NoF/'


# get fish data part
x_data_list = []
scan_wnd_size = [24, 24]
for fish_dir in fish_dirs:
    folder_path = '../cropped train dataset/' + fish_dir + '/'
    fish_pics = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for fish_pic_name in fish_pics:
        image_data = Image.open(folder_path + fish_pic_name)
        image_data = image_data.filter(ImageFilter.FIND_EDGES)
        image_data = resizeimage.resize_cover(image_data, scan_wnd_size, validate=False)
        image_data = image_data.convert('L')
        image_data = numpy.array(image_data, dtype=numpy.float32)
        x_data_list.append(copy.copy(image_data))


x_data = numpy.array(x_data_list)
print x_data.shape
numpy.save(array_save_dir + 'fish_imagedata_%dx%d.npy'%(scan_wnd_size[0], scan_wnd_size[1]), x_data_list)

## no fish part
x_data_list = []
scan_wnd_size = [48, 48]
no_patch_eachpic = 10
nofish_pics = [f for f in listdir(training_fish_dir+no_fish_dir)
               if isfile(join(training_fish_dir+no_fish_dir, f))]

for nofish_pic_name in nofish_pics:
    image = Image.open(training_fish_dir+no_fish_dir+nofish_pic_name)

    for i in range(no_patch_eachpic):
        while True:
            try:
                y = random.randint(0, 720)
                x = random.randint(0, 1280)

                crop_rectangle = (int(x), int(y), int(x + 100), int(y + 100))
                image_data = image.crop(crop_rectangle)
                image_data = image_data.filter(ImageFilter.FIND_EDGES)
                image_data = resizeimage.resize_cover(image_data, scan_wnd_size, validate=False)
                image_data = image_data.convert('L')
                image_data = numpy.array(image_data, dtype=numpy.float32)
                x_data_list.append(copy.copy(image_data))
                break
            except:
                print 'wrong'
                pass

x_data = numpy.array(x_data_list)
print x_data.shape
numpy.save(array_save_dir + 'nofish_imagedata_%dx%d.npy'%(scan_wnd_size[0], scan_wnd_size[1]), x_data_list)

