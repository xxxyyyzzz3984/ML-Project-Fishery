import csv
import os
from os import listdir
from os.path import isfile, join
import copy
import tensorflow as tf
import numpy

def all_same(items):
    return all(x == items[0] for x in items)

all_test_pics = [f for f in listdir('../test dataset/')
                 if isfile(join('../test dataset/', f))]

fish_kinds = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

csv_filename = 'result2.csv'
writer = csv.writer(open(csv_filename, 'wb'))



f = open('result.csv')
for line in f:
    try:
        alread_save_image = line.replace('\n', '').split(',')[0]
        all_test_pics.remove(alread_save_image)
    except:
        pass

remain_images = copy.copy(all_test_pics)

pic_count = 0
for remain_image in remain_images:
    image_info = dict()
    image_info['image'] = remain_image

    saved_images_path = '../find test fish/' + remain_image.replace('.jpg', '') + '/'
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
            from Fish_Classifier.fish_classifier import retrieve_prob_list

            tmp_probs = retrieve_prob_list(image_path)[0]
            tf.reset_default_graph()

            max_indexes.append(numpy.argmax(tmp_probs))
            fish_probs_all.append(copy.copy(tmp_probs))

            # if max_prob <= max(tmp_probs):
            #     max_prob = max(tmp_probs)
            #     fish_probs = copy.copy(tmp_probs)


        if (not all_same(max_indexes) and 1 in max_indexes
            and max(fish_probs_all[max_indexes.index(1)]) > 0.9) and \
                (not all_same(max_indexes) and 6 in max_indexes
            and max(fish_probs_all[max_indexes.index(6)]) > 0.9):

            for max_index in max_indexes:
                if max_index != 1 and max_index != 6:
                    fish_probs = copy.copy(fish_probs_all[max_indexes.index(max_index)])
                    break

            if len(fish_probs) < 1:
                for fish_prob_all in fish_probs_all:
                    if max_prob <= max(fish_prob_all):
                        max_prob = max(fish_prob_all)
                        fish_probs = copy.copy(fish_prob_all)

        else:
            for fish_prob_all in fish_probs_all:
                if max_prob <= max(fish_prob_all):
                    max_prob = max(fish_prob_all)
                    fish_probs = copy.copy(fish_prob_all)


        print fish_kinds[numpy.argmax(fish_probs)] + ' fish'

        for i in range(len(fish_probs)):
            image_info[fish_kinds[i]] = fish_probs[i]

        image_info['NoF'] = float(0.0)

    writer.writerow([image_info['image'], image_info['ALB'], image_info['BET'], image_info['DOL'],
             image_info['LAG'], image_info['NoF'], image_info['OTHER'], image_info['SHARK'],
             image_info['YFT']])

    pic_count += 1
    print 'Finish processing %d fish.' % pic_count