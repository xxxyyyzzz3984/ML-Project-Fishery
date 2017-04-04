import os
import tensorflow as tf
from os.path import isfile, join
import csv
import numpy


dataset_dir = '../test dataset/'
save_csv_path = 'result.csv'
already_pics = []
count = 0
try:
    writer = csv.writer(open(save_csv_path, 'a'))
except IOError:
    writer = csv.writer(open(save_csv_path, 'wb'))
    writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])


f = open(save_csv_path, 'r')
for line in f:
    image_name = line.replace('\n', '').split(',')[0]
    if image_name == 'image':
        continue
    already_pics.append(image_name)

f.close()

all_pics = [f for f in os.listdir(dataset_dir)
                         if isfile(join(dataset_dir, f))]

if len(already_pics) > 0:
    for already_pic in already_pics:
        already_pic = already_pic.replace('+AF8-', '_')
        all_pics.remove(already_pic)

fish_kinds = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

image_info = dict()
for pic in all_pics:
    image_path = dataset_dir + pic

    from Fish_Classifier.fish_classifier import retrieve_prob_list

    fish_probs = retrieve_prob_list(image_path)[0]
    tf.reset_default_graph()

    image_info['image'] = pic

    for i in range(len(fish_probs)):
        image_info[fish_kinds[i]] = fish_probs[i]

    writer.writerow([image_info['image'], image_info['ALB'], image_info['BET'], image_info['DOL'],
                     image_info['LAG'], image_info['NoF'], image_info['OTHER'], image_info['SHARK'],
                     image_info['YFT']])

    count += 1

    print 'Process %d pic with name ' %(count) + pic
    print fish_kinds[numpy.argmax(fish_probs)] + ' fish'

    print


############# change NOF ###############
# save_csv_path = 'result_revised.csv'
# standard_csv = 'result_standard.csv'
# nof_images = []
# other_images = []
# f = open(standard_csv, 'r')
# for line in f:
#     try:
#         nof_prob = line.replace('\n', '').split(',')[5]
#         nof_prob = float(nof_prob)
#         other_prob = line.replace('\n', '').split(',')[6]
#         other_prob = float(other_prob)
#         if nof_prob > 0.3:
#            nof_images.append(line.replace('\n', '').split(',')[0])
#
#         if other_prob > 0.3:
#             other_images.append(line.replace('\n', '').split(',')[0])
#     except ValueError:
#         pass
#
# f.close()
#
# r = csv.reader(open(save_csv_path))
# lines = [l for l in r]
#
# for i in range(len(lines)):
#     #for nof_image in nof_images:
#     #    if lines[i][0] == nof_image:
#     #        lines[i] = [nof_image, '0.04', '0.04', '0.04', '0.04', '0.95', '0.04', '0.04', '0.04']
#
#
#     for other_image in other_images:
#         if lines[i][0] == other_image:
#             lines[i] = [other_image, '0.04', '0.04', '0.04', '0.04', '0.04', '0.95', '0.04', '0.04']
#
#
# writer = csv.writer(open(save_csv_path, 'w'))
# writer.writerows(lines)
#######################################


########### change DOL LAG ###############
# save_csv_path = 'result_revised.csv'
# standard_csv = 'result_standard.csv'
# dol_images = []
# lag_images = []
# f = open(standard_csv, 'r')
# for line in f:
#     try:
#         dol_prob = line.replace('\n', '').split(',')[3]
#         dol_prob = float(dol_prob)
#         lag_prob = line.replace('\n', '').split(',')[4]
#         lag_prob = float(lag_prob)
#         if dol_prob > 0.3:
#             dol_images.append(line.replace('\n', '').split(',')[0])
#
#         if lag_prob > 0.3:
#             lag_images.append(line.replace('\n', '').split(',')[0])
#     except ValueError:
#         pass
#
# f.close()
#
# r = csv.reader(open(save_csv_path))
# lines = [l for l in r]
#
# for i in range(len(lines)):
#     for dol_image in dol_images:
#         if lines[i][0] == dol_image:
#             lines[i] = [dol_image, '0.04', '0.04', '0.9', '0.04', '0.04', '0.04', '0.04', '0.04']
#
#
#     for lag_image in lag_images:
#         if lines[i][0] == lag_image:
#             lines[i] = [lag_image, '0.04', '0.04', '0.04', '0.9', '0.04', '0.04', '0.04', '0.04']
#
#
# writer = csv.writer(open(save_csv_path, 'w'))
# writer.writerows(lines)
#######################################
