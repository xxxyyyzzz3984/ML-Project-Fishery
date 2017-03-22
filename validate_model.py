import numpy
import os
from os.path import isfile, join
import tensorflow as tf

target_dir = '/home/xyh3984/Downloads/extend data/SHARK/'

target_images = [f for f in os.listdir(target_dir)
                         if isfile(join(target_dir, f))]
wnd_size = [220, 220]
total = 0
correct = 0
for target_image in target_images:
    image_path = target_dir + target_image

    print image_path

    from Fish_Classifier.fish_classifier import retrieve_prob_list
    tmp_probs = retrieve_prob_list(image_path)[0]
    tf.reset_default_graph()

    if numpy.argmax(tmp_probs) == 5:
        correct += 1

    total += 1

    print tmp_probs
    print float(correct)/total

print correct
