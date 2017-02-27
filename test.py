import json
from PIL import Image

training_parsing_json_dir = '../dataset parsing/'
cropped_train_dir = '../cropped train dataset/'
training_fish_dir = '../train dataset/'

js_filenames = ['ALB.json', 'BET.json', 'DOL.json', 'LAG.json', 'OTHER.json', 'SHARK.json', 'YFT.json']
# js_filenames = ['ALB.json']
min_w = 100000
min_h = 100000
for js_file in js_filenames:
    js_f = open(training_parsing_json_dir + js_file, 'r')

    lines = ''
    for line in js_f:
        lines += line.replace('\n', '')

    fish_js_data = json.loads(lines)

    for i in range(len(fish_js_data)):
        each_fish_data_js = fish_js_data[i]

        fish_info_list = each_fish_data_js['annotations']
        im = Image.open(training_fish_dir + each_fish_data_js['filename'])
        for fish_info_js in fish_info_list:
            x = fish_info_js['x']
            y = fish_info_js['y']
            width = fish_info_js['width']
            height = fish_info_js['height']

            if width <= min_w and width > 0:
                min_w = width

            if height <= min_h and height > 0:
                min_h = height


print min_w, min_h