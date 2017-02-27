import json
from PIL import Image

training_parsing_json_dir = '../dataset parsing/'
cropped_train_dir = '../cropped train dataset/'
training_fish_dir = '../train dataset/'

js_filenames = ['ALB.json', 'BET.json', 'DOL.json', 'LAG.json', 'OTHER.json', 'SHARK.json', 'YFT.json']

count = 1
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

            crop_rectangle = (int(x), int(y), int(x + width), int(y + height))
            cropped_im = im.crop(crop_rectangle)

            save_dir = cropped_train_dir + js_file.split('.')[0] + '/' + str(count) + '.jpg'
            cropped_im.save(save_dir, 'JPEG')
            count += 1