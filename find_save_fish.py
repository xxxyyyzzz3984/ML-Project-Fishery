import argparse
import os
import tensorflow as tf
from Pretrained_VGG_Fish_Detector.vgg16_fish_detector import find_fish_vgg
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

        else:
            print 'Using Onet Fish Detector'
            from Onet_Fish_Detector.Onet_Detector import find_save_fish_onet
            test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
            find_save_fish_onet(image_path, test_pic_name, save_root_dir)
            tf.reset_default_graph()

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
            pass

        else:
            print 'Using Onet Fish Detector 2'
            from Onet_Fish_Detector.Onet_Detector import find_save_fish_onet
            test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
            find_save_fish_onet(image_path, test_pic_name, save_root_dir)
            tf.reset_default_graph()


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
