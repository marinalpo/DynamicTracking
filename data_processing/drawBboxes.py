import numpy as np
import cv2

frame_path = '/Users/marinaalonsopoal/Desktop/ants1/images/00000001.jpg'
title = 'VOT2018 Sequence ants1'
object_name = 'ant'

ima = cv2.imread(frame_path)

colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.75

init_rects = [(107, 445, 62, 100), (236, 211, 104, 67), (187, 621, 88, 60),
              (225, 561, 90, 67), (166, 444, 83, 58), (233, 455, 60, 93)]

for i in range(len(init_rects)):
    text = object_name + str(i+1)
    x, y, w, h = init_rects[i]
    target_pos = np.array([x + w / 2, y + h / 2])  # Centroids
    cv2.rectangle(ima, (x, y), (x+w, y+h), colors[i], 3)
    cv2.circle(ima, (int(target_pos[0]), int(target_pos[1])), 3, colors[i], -1)
    cv2.putText(ima, text, (x, y-10), font, font_size, colors[i], 2, cv2.LINE_AA)

cv2.imshow(title, ima)

cv2.waitKey(0)
