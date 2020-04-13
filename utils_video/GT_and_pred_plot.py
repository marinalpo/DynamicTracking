# import pandas as pd
import numpy as np
import cv2
import os
from utils_plot import *

# Parameters
sequence = 'acrobats'
num_frames = 40

# Definitions
path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/' + sequence + '/gt/'
img_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/' + sequence + '/img/'
save_path = '/Users/marinaalonsopoal/Desktop/plot/'
columns_standard = ['FrameID',	'ObjectID',	'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'isOccluded']
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.4

# Load data
gt = np.loadtxt(path + 'gt.txt', delimiter=',')  # shape: (n_entries, 8)
init = np.loadtxt(path + 'init.txt', delimiter=',')  # shape: (n_entries, 8)
pred = np.loadtxt(path + 'pred.txt', delimiter=',')  # shape: (n_entries, 8)
img_files = sorted(os.listdir(img_path))[0:num_frames]  # local names 123456.jpg
ims = [cv2.imread(os.path.join(img_path, imf)) for imf in img_files]

num_obj = int(np.max(gt[:, 1]))  # Number of objects that are being tracked in the whole sequence
colors = create_colormap_hsv(num_obj)

for f, im in enumerate(ims):
    print('Frame:', f)
    cv2.putText(im, 'Initialization', (5, 20), font, font_size * 1.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(im, 'Ground Truth', (5, 40), font, font_size*1.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(im, 'Predictions', (5, 60), font, font_size*1.5, (0, 0, 255), 2, cv2.LINE_AA)
    gt_frame = gt[gt[:, 0] == f+1, :]
    gt_frame = gt_frame[gt_frame[:, 6] == 1, :]
    gt_frame = gt_frame[gt_frame[:, 7] == 0, :]
    pred_frame = pred[pred[:, 0] == f+1, :]

    for o in range(gt_frame.shape[0]):
        obj_id = int(gt_frame[o, 1])
        obj_id_pred = int(pred_frame[o, 1])
        location = get_location(gt_frame[o, 2:6])
        loc_pred = get_location(pred_frame[o, 2:6])
        if gt_frame[o, 6] == 1 and gt_frame[o, 7] == 0:
            cv2.polylines(im, [loc_pred], True, (0, 0, 255), 2)
            cv2.polylines(im, [location], True, (0, 255, 0), 2)
            cv2.putText(im, 'ID: ' + str(obj_id_pred), (int(pred_frame[o, 2]), int(pred_frame[o, 3]) - 5), font,
                        font_size, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(im, 'ID: ' + str(obj_id), (int(gt_frame[o, 2]), int(gt_frame[o, 3]) - 5), font,
                        font_size, (0, 255, 0), 1, cv2.LINE_AA)

    init_frame = init[init[:, 0] == f + 1, :]
    for o in range(init_frame.shape[0]):
        obj_id = int(init_frame[o, 1])
        location = get_location(init_frame[o, 2:6])
        cv2.polylines(im, [location], True, (0, 255, 255), 2)
        cv2.putText(im, 'ID: ' + str(obj_id), (int(init_frame[o, 2]), int(init_frame[o, 3]) - 5), font,
                    font_size, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Frame:' + str(f), im)
    cv2.waitKey(0)
    cv2.imwrite(save_path + str(f+1).zfill(6) + '.jpg', im)