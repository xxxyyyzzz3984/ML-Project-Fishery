from skimage import io, transform, filters, color
from skimage.feature import canny
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy


h = 190.0
w = 95.0
x = 651.0
y = 422.0
scan_wnd_size = [256, 256]

image_path = '../train dataset/ALB/img_00010.jpg'

image = io.imread(image_path)

ori_size = image.shape

image_data = transform.resize(image, numpy.array(scan_wnd_size))

x = int(float((x*256)/ori_size[1]))
y = int(float((y*256)/ori_size[0]))

w = int(float((w*256)/ori_size[1]))
h = int(float((h*256)/ori_size[0]))

plt.imshow(image_data)
ax = plt.gca()
rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
ax.add_patch(rect)
plt.show()
