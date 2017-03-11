import commands
import os
from skimage import io, transform
import numpy
import copy


fish_label_strs = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
               'n02607072', 'n02641379', 'n02643566', 'n01496331', 'n02536864', 'n02066245']

prob_thresh = 0.2


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

def find_fish_vgg(image_path, image_name, save_root_dir):

    has_fish = False

    image = io.imread(image_path)

    test_pic_path = image_path.replace(' ', '\ ')

    search_stride = 100

    i_w = 400
    j_w = 400

    i_total = image.shape[0] / search_stride
    j_total = image.shape[1] / search_stride
    count = 1

    for i in range(i_total):

        print 'Performin sliding window detection'


        command = 'python Pretrained_VGG_Fish_Detector/vgg16_findfish_slidewnd.py' \
                  ' -i %s -ti %d -ss %d -w %d' % \
                  (test_pic_path, i, search_stride, j_w)

        results_raw_list = commands.getstatusoutput(command)[1].split('\n')
        print results_raw_list
        labels = []
        probs = []
        for result_raw in results_raw_list:
            if 'Top1' in result_raw:
                result_parts = result_raw.split(',')
                label = result_parts[1][2:11]
                prob = float(result_parts[len(result_parts) - 1].replace(')', ''))

                labels.append(label)
                probs.append(prob)

        print j_total
        for j in range(j_total):
            try:
                image_data = copy.copy(image[j * search_stride:j * search_stride + j_w,
                                       i * search_stride:i * search_stride + i_w, 0:3])
                transform.resize(image_data, [224, 224])

                label = labels[j]
                prob = probs[j]

                if label in fish_label_strs:
                    if prob >= prob_thresh:
                        mkdir_save_root_dir = save_root_dir.replace(' ', '\ ')
                        os.system('mkdir ' + mkdir_save_root_dir + image_name + '/')
                        io.imsave(save_root_dir + image_name + '/' + str(count) + '.jpg', image_data)
                        count += 1
                        has_fish = True

            except:
                continue

    return has_fish


