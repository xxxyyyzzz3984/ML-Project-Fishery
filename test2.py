import numpy
from skimage import data, color, exposure, filters, io, transform
import struct,array

import cv2
import numpy as np

def showvec(fn, width=24, height=24, resize=4.0):
  f = open(fn,'rb')
  HEADERTYP = '<iihh' # img count, img size, min, max

  # read header
  imgcount,imgsize,_,_ = struct.unpack(HEADERTYP, f.read(12))

  for i in range(imgcount):
    img  = np.zeros((height,width),np.uint8)

    f.read(1) # read gap byte

    data = array.array('h')

    ###  buf = f.read(imgsize*2)
    ###  data.fromstring(buf)

    data.fromfile(f,imgsize)

    for r in range(height):
      for c in range(width):
        img[r, c] = data[r * width + c]

    # img = cv2.resize(img, (0,0), fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    image_data = transform.resize(img, numpy.array([48, 48]))

    cv2.imshow('vec_img', image_data)
    k = 0xFF & cv2.waitKey(0)
    if k == 27:         # esc to exit
      break


showvec('../array train dataset/vec_fish_positive/fish_all_40_30.vec', 40, 30)