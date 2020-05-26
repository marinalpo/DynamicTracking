import numpy as np
import cv2
import scipy.misc



x = np.load('/Users/marinaalonsopoal/Desktop/patch.npy')
# x = x.squeeze(0)
# x = np.transpose(x, (1, 2, 0))
#
d = np.load('/Users/marinaalonsopoal/Desktop/delta2.npy')
num_boxes = d.shape[1]
print(d)

for i in range(num_boxes):
    x1 = d[0, i] + 255/2
    y1 = d[1, i] + 255/2
    x2 = d[2, i] + 255/2
    y2 = d[3, i] + 255/2

    # print(x1, y1, x2, y2)
    # if i % 51 == 0:
    # cv2.rectangle(x, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)

fraction = 255/25
print(fraction)
for j in range(0, 26):
    cv2.line(x, (0, int(j*fraction)), (255, int(j*fraction)), (0, 255, 255))
    cv2.line(x, (int(j * fraction), 0), (int(j * fraction), 255), (0, 255, 255))


cv2.imshow('deltas', x)
cv2.waitKey()

