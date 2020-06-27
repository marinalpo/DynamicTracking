import cv2
import numpy as np
import pickle as pkl
import os
import glob



num_end = 89
sequence = 'football'
img_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/eSMOT/'+str(sequence)+'/img/'
green = (0, 255, 0)
red = (0, 0, 255)

with open('/Users/marinaalonsopoal/Desktop/pred_loc_siam.obj', 'rb') as f:
    pred_loc_siam = pkl.load(f)

with open('/Users/marinaalonsopoal/Desktop/pred_loc_ours.obj', 'rb') as f:
    pred_loc_ours = pkl.load(f)


img_files = sorted(glob.glob(os.path.join(img_path, '*.jp*')))[000000:num_end]
ims = [cv2.imread(imf) for imf in img_files]


for f, im in enumerate(ims):
    if f in pred_loc_siam.keys():
        loc_siam = pred_loc_siam[f]
        loc_ours = pred_loc_ours[f]
        print('f:', f)
        cv2.polylines(im, [loc_siam], True, red, 8)
        cv2.polylines(im, [loc_ours], True, green, 8)
        cv2.imwrite('/Users/marinaalonsopoal/Desktop/hockey/' + str(f).zfill(6) + '.jpg', im)