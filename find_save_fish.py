import argparse
import os
import tensorflow as tf
import time

from Pretrained_VGG_Fish_Detector.vgg16_fish_detector import find_fish_vgg
from Pretrained_VGG_Fish_Detector.vgg16_findfish_detector_slidewnd import find_fish_vgg_slidewnd
from os import listdir
from os.path import isfile, join

def wrapup_save_fish(image_path, image_name, save_root_dir):
    print 'Processing image ' + image_name
    print 'Using VGG Fish Detector'
    test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
    has_fish_vgg = find_fish_vgg(image_path, test_pic_name, save_root_dir)

    if not has_fish_vgg:
        print 'Using NOF Fish Detector'
        from NoF_Detector.NoF_Detector import predict_has_fish
        has_fish_nof = predict_has_fish(image_path)
        tf.reset_default_graph()

        if not has_fish_nof:
            print 'NoF does not find fish, skip'
            save_root_sys_dir = save_root_dir.replace(' ', '\ ')
            os.system('mkdir ' + save_root_sys_dir + test_pic_name + '/')

        else:
            print 'Using Onet Fish Detector'
            from Onet_Fish_Detector.Onet_Detector import find_save_fish_onet
            test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
            has_fish_onet = find_save_fish_onet(image_path, test_pic_name, save_root_dir)
            tf.reset_default_graph()
            if not has_fish_onet:
                find_fish_vgg_slidewnd(image_path, test_pic_name, save_root_dir)


    else:
        vgg_find_imgs = [f for f in listdir(save_root_dir + test_pic_name)
                         if isfile(join(save_root_dir + test_pic_name, f))]

        for vgg_find_img in vgg_find_imgs:
            from Onet_Fish_Detector.Onet_Detector import pre_screen_is_fish

            vgg_find_imgpath = save_root_dir + test_pic_name + '/' + vgg_find_img
            is_fish = pre_screen_is_fish(vgg_find_imgpath)
            tf.reset_default_graph()

            if not is_fish:
                print 'false fish, deleting...'
                os.remove(vgg_find_imgpath)

        vgg_find_imgs = [f for f in listdir(save_root_dir + test_pic_name)
                         if isfile(join(save_root_dir + test_pic_name, f))]

        # no fish left
        if len(vgg_find_imgs) < 1:
            print 'Using NOF Fish Detector 2'
            from NoF_Detector.NoF_Detector import predict_has_fish
            has_fish_nof = predict_has_fish(image_path)
            tf.reset_default_graph()

            if not has_fish_nof:
                print 'Really no fish, Skip!'
                save_root_sys_dir = save_root_dir.replace(' ', '\ ')
                os.system('mkdir ' + save_root_sys_dir + test_pic_name + '/')

            else:
                print 'Using Onet Fish Detector 2'
                from Onet_Fish_Detector.Onet_Detector import find_save_fish_onet
                test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
                has_fish_onet = find_save_fish_onet(image_path, test_pic_name, save_root_dir)
                tf.reset_default_graph()
                if not has_fish_onet:
                    find_fish_vgg_slidewnd(image_path, test_pic_name, save_root_dir)


if __name__ == '__main__':

    # ap = argparse.ArgumentParser()
    # ap.add_argument('-f', nargs=1)
    #
    # opts = ap.parse_args()
    # images_folder = opts.f[0]
    #
    # test_pics = [f for f in listdir(images_folder)
    #              if isfile(join(images_folder, f))]
    #
    # count = 0
    # for test_pic in test_pics:
    #     while True:
    #         try:
    #             wrapup_save_fish(images_folder + test_pic, test_pic, '../find test fish/')
    #             break
    #         except:
    #             time.sleep(10)

    all_test_pics = [f for f in listdir('../test dataset/')
                 if isfile(join('../test dataset/', f))]

    already_saved_images = [x[0] for x in os.walk('../find test fish/')]

    not_saved_images = []
    for already_saved_image in already_saved_images:
        already_saved_image = already_saved_image.replace('../find test fish/', '')
        if already_saved_image == '':
            continue

        already_saved_image = already_saved_image + '.jpg'

        all_test_pics.remove(already_saved_image)

    for all_test_pic in all_test_pics:
        os.system('mkdir ' + '../find\ test\ fish/' + all_test_pic.replace('.jpg', ''))

    # f = open('record_fish.txt', 'a')
    # for all_test_pic in all_test_pics:
    #     f.write(all_test_pic)
    #     f.write('\n')
    #
    # f.close()

    # ap = argparse.ArgumentParser()
    # ap.add_argument('-f', nargs=1)
    #
    # opts = ap.parse_args()
    # record_file = opts.f[0]
    #
    # with open(record_file, 'r') as f:
    #     for line in f:
    #         test_pic = line.replace('\n', '')
    #         while True:
    #             try:
    #                 wrapup_save_fish('../test dataset/' + test_pic, test_pic, '../find test fish/')
    #                 break
    #             except:
    #                 time.sleep(10)
    #
    #     f.close()

                    # for test_pic in all_test_pics:
    #     while True:
    #         try:
    #             wrapup_save_fish('../test dataset/' + test_pic, test_pic, '../find test fish/')
    #             break
    #         except:
    #             time.sleep(10)