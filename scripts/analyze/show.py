import numpy as np
import cv2

import sys
klass = 1
if len(sys.argv) > 2:
    _, path, klass = sys.argv
    klass = int(klass)
else:
    _, path = sys.argv
data = np.loadtxt(path, delimiter=',')
data = data[data[:,0]==klass,1:]
np.random.shuffle(data)

n_rand = 100
n_hor = 10
n_ver = int(np.ceil(100/10))

width = 24
height = 12

data = data[:n_rand,:]

pic_picture = np.zeros((n_ver * (height+1) + 1, n_hor * (width+1)+1, 3), dtype=np.uint8)
for y in xrange(n_ver):
    for x in xrange(n_hor):
        pic = data[y*n_hor+x,:]
        # convert to rgb
        pic_r = pic[0::3]
        pic_g = pic[1::3]
        pic_b = pic[2::3]
        pic = np.dstack([pic_r.reshape((height,width)), pic_g.reshape((height, width)), pic_b.reshape((height, width))])
        pic = pic.astype(np.uint8)

        y_start = (height+1)*y + 1
        x_start = (width+1)*x + 1
        y_end = y_start + height
        x_end = x_start + width

        pic_picture[y_start:y_end, x_start:x_end,:] = pic

cv2.imshow("DA PIC", pic_picture)
cv2.waitKey(0)
