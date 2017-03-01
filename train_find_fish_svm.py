import Image
import ImageFilter

from resizeimage import resizeimage
from sklearn import svm
import numpy
import copy
import pickle

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn.externals import joblib

scan_wnd_size = [16, 16]
y_data_list = []

with open('../array train dataset/fish_hog_100x100.pkl', 'rb') as fp:
    x_data_list = pickle.load(fp)

fish_len = len(x_data_list)

with open('../array train dataset/nofish_hog_100x100.pkl', 'rb') as fp:
    x_data_list += pickle.load(fp)

for i in range(len(x_data_list)):
    if i < fish_len:
        y_data_list.append(1) ## denotes has fish
    else:
        y_data_list.append(0) ## denotes no fish



clf = svm.SVR()
clf.fit(x_data_list, y_data_list)

joblib.dump(clf, 'svr.pkl')

# clf = joblib.load('svr.pkl')
#
# im_test = Image.open('2.png')
# im_test = im_test.filter(ImageFilter.FIND_EDGES)
# im_test = resizeimage.resize_cover(im_test, scan_wnd_size, validate=False)
# im_test = im_test.convert('L')
# im_test = numpy.array(im_test, dtype=numpy.float32)
# fd, hog_image = hog(im_test, orientations=8, pixels_per_cell=(16, 16),
#                                     cells_per_block=(1, 1), visualise=True)
#
# hog_image = hog_image.reshape(scan_wnd_size[0]*scan_wnd_size[1], )
# hog_image = hog_image.tolist()
#
#
# # clf.predict([[2., 2.]])
#
# print clf.predict([hog_image])