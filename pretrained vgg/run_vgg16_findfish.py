import commands
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage import io
import matplotlib.patches as mpatches
import numpy


test_pics_folder = '../../train dataset/BET/'

test_pics = [f for f in listdir(test_pics_folder)
               if isfile(join(test_pics_folder, f))]

fish_label_strs = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
               'n02607072', 'n02641379', 'n02643566', 'n01496331']

prob_thresh = 0.3


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = numpy.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last],
                                                     numpy.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

for test_pic_name in test_pics:
    test_pic_path = test_pics_folder + test_pic_name

    image = io.imread(test_pic_path)

    test_pic_path = test_pic_path.replace(' ', '\ ')

    search_stride = 100

    i_w = 400
    j_w = 400

    i_total = image.shape[0] / search_stride
    j_total = image.shape[1] / search_stride
    plt.imshow(image)
    ax = plt.gca()
    max_prob = 0
    rect = None

    boxes = []

    for i in range(i_total):

        run_command = 'python vgg16_findfish.py -i %s -si %d -ei %d -sj %d -ej %d -w %d -ss %d' % (
        test_pic_path, i, i+1, 0, j_total, i_w, search_stride)
        results_raw = commands.getstatusoutput(run_command)

        results_str = results_raw[1]
        results_str = results_str.split('\n')
        j = 0

        for result_str in results_str:
            result_str = result_str.replace('\n', '')
            if 'Top' not in result_str:
                continue

            result_parts = result_str.split(',')
            label = result_parts[1][2:11]
            prob = float(result_parts[len(result_parts)-1].replace(')', ''))

            print label
            print prob

            j += 1

            if label in fish_label_strs:
                if prob >= prob_thresh:
                    boxes.append([j * search_stride, i * search_stride,
                                  j * search_stride + j_w, i * search_stride + i_w])


    boxes_arr = numpy.array(boxes)
    boxes_arr = non_max_suppression_fast(boxes_arr, 0.4)
    #
    for box_arr in boxes_arr:
        rect = mpatches.Rectangle((box_arr[0], box_arr[1]), box_arr[2] - box_arr[0],
                                  box_arr[3] - box_arr[1],
                                  fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)

    plt.show()
