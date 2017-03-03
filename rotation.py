from skimage import data, color, exposure, filters, io, transform, feature
import cv2
import numpy as np

image1 = io.imread('../cropped train dataset/ALB/1.jpg')
image2 = io.imread('../cropped train dataset/ALB/4.jpg')

image1 = color.rgb2grey(image1)
image2 = color.rgb2grey(image2)


image1 = transform.resize(image1, np.array([400, 400]))
image2 = transform.resize(image2, np.array([400, 400]))

angle = 0
max_sim = 0

image1_corners = feature.corner_harris(image1)




for i in range(361):
    image2_rot = transform.rotate(image2, angle=i)
    sim = feature.match_template(image2_rot, image1)

    image2_corners = feature.corner_harris(image2_rot)
    match = feature.match_descriptors(image1_corners, image2_corners)

    print match

    if sim>max_sim:
        max_sim = sim
        angle = i


image2 = transform.rotate(image2, angle=angle)

cv2.imshow('test', image1)
cv2.imshow('test1', image2)
cv2.waitKey()


