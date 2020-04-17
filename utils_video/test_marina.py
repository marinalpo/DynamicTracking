import numpy as np
import cv2
import time

from tqdm import tqdm

image_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/acrobats/img/000005.jpg'
ima = cv2.imread(image_path)
mask = np.load('/Users/marinaalonsopoal/Desktop/mask.npy')

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.6
color = (255, 255, 0)

location = [960.8967, 469.46042, 934.38556, 425.40518, 1079.5969, 338.02127, 1106.1082, 382.0765]

x1 = int(location[0])
y1 = int(location[1])

x2 = int(location[2])
y2 = int(location[3])

x3 = int(location[4])
y3 = int(location[5])

x4 = int(location[6])
y4 = int(location[7])

mask_bin = mask > 0.35
ima[:, :, 2] = (mask_bin > 0) * 255 + (mask_bin == 0) * ima[:, :, 2]

cv2.polylines(ima, [np.int0(location).reshape((-1, 1, 2))], True, color, 1)

cv2.circle(ima, (x1, y1), 3, color, -1)
cv2.putText(ima, '1', (x1, y1-10), font, font_size, color, 2, cv2.LINE_AA)

cv2.circle(ima, (x2, y2), 3, color, -1)
cv2.putText(ima, '2', (x2, y2-10), font, font_size, color, 2, cv2.LINE_AA)

cv2.circle(ima, (x3, y3), 3, color, -1)
cv2.putText(ima, '3', (x3, y3-10), font, font_size, color, 2, cv2.LINE_AA)

cv2.circle(ima, (x4, y4), 3, color, -1)
cv2.putText(ima, '4', (x4, y4-10), font, font_size, color, 2, cv2.LINE_AA)

# cv2.imshow('Location', ima)
# # cv2.imshow('Mascara', mask)
# cv2.waitKey(0)
for i in tqdm(range(1000)):
    time.sleep(0.01)
