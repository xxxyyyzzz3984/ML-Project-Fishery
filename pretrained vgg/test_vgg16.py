from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import cv2, numpy as np
from keras import backend as K
from os import listdir
from os.path import isfile, join
import copy
import matplotlib.patches as mpatches

from keras.optimizers import SGD
from skimage import io
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

fish_labels = [1, 2, 3, 4, 5, 6, 7, 108, 124, 125, 148, 328, 394, 396, 394, 398]

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

test_pics_folder = '../../test dataset/'
test_pics = [f for f in listdir(test_pics_folder)
               if isfile(join(test_pics_folder, f))]

scan_wnd_size = (224, 224)
model = VGG_16('../../saved models/vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

im = cv2.resize(cv2.imread('../../small test/8.png'), (224, 224)).astype(np.float32)
im[:, :, 0] -= 103.939
im[:, :, 1] -= 116.779
im[:, :, 2] -= 123.68
im = im.transpose((2, 0, 1))
im = np.expand_dims(im, axis=0)

# Test pretrained model
# model = VGG_16('vgg16_weights.h5')
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(im)
print np.argmax(out)

# if __name__ == "__main__":
#     for test_pic_name in test_pics:
#         test_pic_path = test_pics_folder + test_pic_name
#
#         image = io.imread(test_pic_path)
#
#
#         search_stride = 300
#
#         i_w = 300
#         j_w = 300
#
#         i_total = image.shape[0] / search_stride
#         j_total = image.shape[1] / search_stride
#         plt.imshow(image)
#         ax = plt.gca()
#         for i in range(i_total):
#             for j in range(j_total):
#
#                 rect = mpatches.Rectangle((j * search_stride, i * search_stride), j_w, i_w,
#                                           fill=False, edgecolor='green', linewidth=2)
#                 ax.add_patch(rect)
#
#                 image_data = None
#
#                 image_data = copy.copy(image[i * search_stride:i * search_stride + i_w,
#                                        j * search_stride:j * search_stride + j_w, 0:3])
#
#                 image_data = cv2.resize(image_data, scan_wnd_size).astype(np.float32)
#                 image_data[:, :, 0] -= 103.939
#                 image_data[:, :, 1] -= 116.779
#                 image_data[:, :, 2] -= 123.68
#                 image_data = image_data.transpose((2, 0, 1))
#                 image_data = np.expand_dims(image_data, axis=0)
#
#
#                 out = model.predict(image_data)
#                 result = np.argmax(out)
#
#                 if result in fish_labels:
#                     rect = mpatches.Rectangle((j * search_stride, i * search_stride), j_w, i_w,
#                                               fill=False, edgecolor='red', linewidth=2)
#                     ax.add_patch(rect)
#
#
#         plt.show()