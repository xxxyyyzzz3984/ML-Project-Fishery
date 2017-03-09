from os import listdir
from os.path import join, isfile
import tensorflow as tf
import time

from Pretrained_VGG_Fish_Detector.vgg16_fish_detector import find_fish_vgg
import threading

def wrapup_save_fish(image_path, image_name, save_root_dir):
    test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
    has_fish_vgg = find_fish_vgg(image_path, test_pic_name, save_root_dir)

    if not has_fish_vgg:
        from NoF_Detector.NoF_Detector import predict_has_fish
        has_fish_nof = predict_has_fish(test_pics_folder + test_pic)
        tf.reset_default_graph()

        if not has_fish_nof:
            pass

        else:
            from Onet_Fish_Detector.Onet_Detector import find_fish_onet
            test_pic_name = image_name.replace('.jpg', '').replace('.png', '')
            find_fish_onet(image_path, test_pic_name, save_root_dir)


if __name__ == '__main__':
    test_pics_folder = '../test dataset/'

    test_pics = [f for f in listdir(test_pics_folder)
                 if isfile(join(test_pics_folder, f))]

    count = 0
    t = [None, None, None, None, None]
    for test_pic in test_pics:
        t[count] = threading.Thread(target=wrapup_save_fish,
            args=(test_pics_folder + test_pic, test_pic, '../find test fish/'))

        t[count].start()

        count += 1

        if count == 5:
            t[0].join()
            t[1].join()
            t[2].join()
            t[3].join()
            t[4].join()

            count = 0




