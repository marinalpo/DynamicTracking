import numpy as np
import cv2
import os
from utils_plot import *


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
dataset = 1  # 0: MOT, 1: SMOT, 2: Stanford
num_frames = 50
root = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets'

if dataset == 0:  # 2DMOT2015
    title = 'Dataset: 2DMOT2015 Sequence: PETS09-S2L1'
    img_path = root + '/2DMOT2015/train/PETS09-S2L1/img1/'
    gt_path = root + '/2DMOT2015/train/PETS09-S2L1/gt/gt_MOT_pr.txt'
    save_path = root + '/2DMOT2015/train/PETS09-S2L1/gt_plot/'
elif dataset == 1:  # SMOT
    sequence = 'seagulls'
    title = 'Dataset: SMOT Sequence: ' + sequence
    img_path = root + '/SMOT/' + sequence + '/img/'
    gt_path = root + '/SMOT/' + sequence + '/gt/gt.txt'
    save_path = root + '/SMOT/' + sequence + '/gt/gt_plot/'
else:  # Stanford
    title = 'Dataset: StanfordCampus Sequence: Bookstore (Video 0)'
    img_path = root + '/stanford_campus/videos/bookstore/frames0/'
    gt_path = root + '/stanford_campus/annotations/bookstore/video0/gt_stanford_pr.txt'
    save_path = root + '/stanford_campus/annotations/bookstore/video0/plot_GT/'


# Load Data
# ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', ''Height', 'isActive', 'isOccluded']
gt = np.loadtxt(gt_path, delimiter=',')  # shape: (n_entries, 8)
img_files = sorted(os.listdir(img_path))[0:num_frames]  # local names 123456.jpg
ims = [cv2.imread(os.path.join(img_path, imf)) for imf in img_files]

num_obj = int(np.max(gt[:, 1]))  # Number of objects that are being tracked in the whole sequence
colors = create_colormap_hsv(num_obj)

for f, im in enumerate(ims):
    gt_frame = gt[gt[:, 0] == f+1, :]
    for o in range(gt_frame.shape[0]):
        obj_id = int(gt_frame[o, 1])
        location = get_location(gt_frame[o, 2:6])
        if gt_frame[o, 6] == 1 and gt_frame[o, 7] == 0:
            cv2.polylines(im, [location], True, colors[obj_id-1], 2)
            cv2.putText(im, 'ID: ' + str(obj_id), (int(gt_frame[o, 2]), int(gt_frame[o, 3]) - 5), font,
                        font_size, colors[obj_id-1], 1, cv2.LINE_AA)
    cv2.imshow(title + 'Frame:' + str(f), im)
    cv2.waitKey(0)
    cv2.imwrite(save_path + str(f+1).zfill(6) + '.jpg', im)
