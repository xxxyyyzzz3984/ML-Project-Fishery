import argparse
from Pretrained_VGG_Fish_Detector.vgg16_findfish_detector_slidewnd import find_fish_vgg
from os import listdir, system
from os.path import isfile, join

def wrapup_save_fish(image_path, image_name, save_root_dir):
    print 'Processing image ' + image_name
    print 'Using VGG Fish Detector'
    test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
    has_fish_vgg = find_fish_vgg(image_path, test_pic_name, save_root_dir)
    if not has_fish_vgg:
        system('mkdir ' + save_root_dir + test_pic_name + '/')


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-f', nargs=1)

    opts = ap.parse_args()
    images_folder = opts.f[0]

    test_pics = [f for f in listdir(images_folder)
                 if isfile(join(images_folder, f))]

    count = 0
    for test_pic in test_pics:
        wrapup_save_fish(images_folder + test_pic, test_pic, '../find test fish/')
