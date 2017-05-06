import os
import tensorflow as tf
from os.path import isfile, join
import copy
import csv
import numpy
import cv2

def all_same(items):
    return all(x == items[0] for x in items)

fish_kinds = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

csv_filename = 'result.csv'
writer = csv.writer(open(csv_filename, 'wb'))

writer.writerow(['image', 'ALB'	, 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])

dataset_dir = 'test dataset/'

saved_images_paths = [x[0] for x in os.walk(dataset_dir)]
saved_images_paths = saved_images_paths[1:len(saved_images_paths)]
pic_count = 0
for saved_images_path in saved_images_paths:
    image_info = dict()
    image_info['image'] = saved_images_path.replace(dataset_dir, '') + '.jpg'

    saved_images_path = saved_images_path + '/'
    pics_of_one_image = [f for f in os.listdir(saved_images_path)
                         if isfile(join(saved_images_path, f))]

    print 'Processing image ' + image_info['image']

    if len(pics_of_one_image) < 1:
        image_info['ALB'] = float(0.0)
        image_info['BET'] = float(0.0)
        image_info['DOL'] = float(0.0)
        image_info['LAG'] = float(0.0)
        image_info['NoF'] = float(1.0)
        image_info['OTHER'] = float(0.0)
        image_info['SHARK'] = float(0.0)
        image_info['YFT'] = float(0.0)
        print 'NoF Fish'

    else:
        fish_probs = []
        fish_probs_all = []
        max_indexes = []
        max_prob = 0
        for pic_per_image in pics_of_one_image:
            image_path = saved_images_path + pic_per_image

            # cv2.imshow("test", cv2.imread(image_path))
            # cv2.waitKey()

            from Fish_Classifier.fish_classifier import retrieve_prob_list

            tmp_probs = retrieve_prob_list(image_path)[0]
            tf.reset_default_graph()

            print tmp_probs
            max_indexes.append(numpy.argmax(tmp_probs))
            fish_probs_all.append(copy.copy(tmp_probs))

            if max_prob <= max(tmp_probs):
                max_prob = max(tmp_probs)
                fish_probs = copy.copy(tmp_probs)


        # if (not all_same(max_indexes) and 1 in max_indexes
        #     and max(fish_probs_all[max_indexes.index(1)]) > 0.9) and \
        #         (not all_same(max_indexes) and 6 in max_indexes
        #     and max(fish_probs_all[max_indexes.index(6)]) > 0.9):
        #
        #     for max_index in max_indexes:
        #         if max_index != 1 and max_index != 6:
        #             fish_probs = copy.copy(fish_probs_all[max_indexes.index(max_index)])
        #             break
        #
        #     if len(fish_probs) < 1:
        #         for fish_prob_all in fish_probs_all:
        #             if max_prob <= max(fish_prob_all):
        #                 max_prob = max(fish_prob_all)
        #                 fish_probs = copy.copy(fish_prob_all)
        #
        # else:
        #     for fish_prob_all in fish_probs_all:
        #         if max_prob <= max(fish_prob_all):
        #             max_prob = max(fish_prob_all)
        #             fish_probs = copy.copy(fish_prob_all)


        print fish_kinds[numpy.argmax(fish_probs)] + ' fish'

        for i in range(len(fish_probs)):
            image_info[fish_kinds[i]] = fish_probs[i]

        image_info['NoF'] = float(0.0)

    writer.writerow([image_info['image'], image_info['ALB'], image_info['BET'], image_info['DOL'],
             image_info['LAG'], image_info['NoF'], image_info['OTHER'], image_info['SHARK'],
             image_info['YFT']])

    pic_count += 1
    print 'Finish processing %d fish.' % pic_count
