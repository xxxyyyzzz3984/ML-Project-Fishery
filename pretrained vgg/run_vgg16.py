import commands
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage import io
import matplotlib.patches as mpatches


test_pics_folder = '../../train dataset/BET/'

test_pics = [f for f in listdir(test_pics_folder)
               if isfile(join(test_pics_folder, f))]

fish_label_strs = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
               'n02607072', 'n02641379', 'n02643566']

prob_thresh = 0.6


for test_pic_name in test_pics:
    test_pic_path = test_pics_folder + test_pic_name

    image = io.imread(test_pic_path)

    test_pic_path = test_pic_path.replace(' ', '\ ')

    search_stride = 100

    i_w = 300
    j_w = 300

    i_total = image.shape[0] / search_stride
    j_total = image.shape[1] / search_stride
    plt.imshow(image)
    ax = plt.gca()
    max_prob = 0
    rect = None

    for i in range(i_total):

        run_command = 'python test_vgg16.py -i %s -si %d -ei %d -sj %d -ej %d -w %d -ss %d' % (
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
                    rect = mpatches.Rectangle((j * search_stride, i * search_stride), j_w, i_w,
                                              fill=False, edgecolor='red', linewidth=2)

                    ax.add_patch(rect)

    plt.show()
