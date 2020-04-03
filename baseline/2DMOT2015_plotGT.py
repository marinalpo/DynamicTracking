import numpy as np
import cv2
import os
from utils_visualization import create_colormap_hsv


def get_location(gt):
    x, y, w, h = gt
    p1 = [x, y]
    p2 = [x + w, y]
    p3 = [x + w, y + h]
    p4 = [x, y + h]
    location = np.concatenate([p1, p2, p3, p4])
    location = np.int0(location).reshape((-1, 1, 2))  # shape (4, 1, 2)
    return location


# Definitions
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.4


# Parameters
img_path = '/Users/marinaalonsopoal/Desktop/2DMOT2015/train/PETS09-S2L1/img1/'
gt_path = '/Users/marinaalonsopoal/Desktop/2DMOT2015/train/PETS09-S2L1/gt/gt.txt'
save_path = '/Users/marinaalonsopoal/Desktop/2DMOT2015/train/PETS09-S2L1/bboxes/'
num_frames = 2

# Load Data
gt = np.loadtxt(gt_path, delimiter=',')  # shape: (n_entries, 10)
img_files = sorted(os.listdir(img_path))[0:num_frames]  # local names 123456.jpg
ims = [cv2.imread(os.path.join(img_path, imf)) for imf in img_files]


gt = gt[:, :-3]   # (Frame ID, Object ID, x_topleft, y_topleft, Width, Height, Entry Flag)
num_obj = int(np.max(gt[:, 1]))  # Number of objects that are being tracked in the whole sequence
print('num obj: ', num_obj)
colors = create_colormap_hsv(num_obj)

for f, im in enumerate(ims):
    print('Frame:', f+1)
    gt_frame = gt[gt[:, 0] == f+1, :]  # shape: (number of objects per frame, 7)
    for o in range(gt_frame.shape[0]):
        obj_id = int(gt_frame[o, 1])
        print('Object ID:', obj_id)
        location = get_location(gt_frame[o, 2:6])
        cv2.polylines(im, [location], True, colors[obj_id-1], 2)
        cv2.putText(im, 'ID: ' + str(obj_id), (int(gt_frame[o, 2]), int(gt_frame[o, 3]) - 5), font, font_size, colors[obj_id-1], 1, cv2.LINE_AA)
    cv2.imshow('Dataset: 2DMOT2015 Sequence: PETS09-S2L1 Frame:' + str(f), im)
    cv2.waitKey(0)
    #cv2.imwrite(save_path + str(f) + '.jpeg', im)
