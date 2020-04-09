import glob
import numpy as np
import cv2
import pickle
import argparse
import pandas as pd
import torch
import json
from tools.test_2_nostre import *
from utils.bbox_helper import get_aligned_bbox
from shapely.geometry import asPolygon
from custom import Custom


def append_pred(pred, frame, objectID, x, y, w, h):
    my_dict = {'FrameID': frame, 'ObjectID': objectID, 'x_topleft': x, 'y_topleft': y, 'Width': w, 'Height': h,
              'isActive': 1, 'isOccluded': 0}
    pred = pred.append(pd.DataFrame(my_dict, index=[0]))
    return pred


def create_colormap_hsv(num_col):
    colors = []
    addGrayScale = False
    rounds = 3
    np.random.seed(2)
    if num_col <= 8:
        rounds = 1
        div = num_col
    elif num_col <= 20:
        div = 7
    else:
        div = int(np.ceil(num_col / 3))
        addGrayScale = True
    if addGrayScale:
        colors.append((220, 220, 220))
        colors.append((20, 20, 20))
        colors.append((127, 127, 127))
    portion = 180 / div
    ss = [255, 255, 100]
    vs = [255, 100, 255]
    for j in range(rounds):
        s = ss[j]
        v = vs[j]
        for i in range(div):
            h = int(portion * i)
            color_hsv = np.uint8([[[h, s, v]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            color_bgr = color_bgr[0][0]
            color_bgr = [int(cha) for cha in color_bgr]
            colors.append(tuple(color_bgr))
    colors = colors[0:num_col]
    np.random.shuffle(colors)
    return colors


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', type=str, metavar='PATH',
                    default='SiamMask_DAVIS.pth', help='path to latest checkpoint')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
args = parser.parse_args()

# PARAMETERS
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
columns_standard = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'isOccluded']

# Video Parameters
draw_mask = False
draw_candidates = False
filter_boxes = True
num_frames = 20
bbox_type = 0, 1  # 0: rotated, 1: aligned

sequence = 'acrobats'
title = 'Dataset: SMOT Sequence:' + sequence
save_data_path = '/data/results2/'
base_path = '/data/SMOT/' + sequence + '/img/'
save_centroids_path = '/data/Marina/centroids/centroids_' + sequence + '.obj'
init_path = '/data/SMOT/' + sequence + '/gt/init.txt'

init = pd.read_csv(init_path, sep=',', header=None)
# Assign column names
init.columns = columns_standard
pred = init.copy()
total_obj = init.ObjectID.max()  # [5, 34, 65, 2...]
colors = create_colormap_hsv(total_obj)

if __name__ == '__main__':
    toc = 0  # Timer

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)
    siammask.eval().to(device)

    # Parse Image files
    img_files = sorted(glob.glob(join(base_path, '*.jp*')))[000:num_frames]
    ims = [cv2.imread(imf) for imf in img_files]
    im_ori = ims[0]  # First frame

    all_centroids = {}  # Dict that will save all the objects trajectories and discarded candidates
    objects = {}

    for f, im in enumerate(ims):
        print('Frame:', f + 1)
        cv2.putText(im, title, (10, 30), font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im, 'Frame: ' + str(f), (10, 60), font, font_size * 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        tic = cv2.getTickCount()

        # Get number of objects that are initialized in this frame
        init_frame = init[init.FrameID == f + 1]
        for index, row in init_frame.iterrows():
            x = row['x_topleft']
            y = row['y_topleft']
            w = row['Width']
            h = row['Height']
            nested_obj = {'target_pos': np.array([x + w / 2, y + h / 2]),
                          'target_sz': np.array([w, h]),
                          'init_frame': f}
            state = siamese_init(im, nested_obj['target_pos'], nested_obj['target_sz'], siammask, cfg['hp'],
                                 device=device)
            nested_obj['state'] = state
            objects[int(row['ObjectID'])] = nested_obj
            cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), colors[int(row['ObjectID']) - 1], 5)

        if f > 0:  # Perform tracking
            for key, value in objects.items():
                frame_boxes = []
                state_init = siamese_init(ims[value['init_frame']], value['target_pos'], value['target_sz'],
                                          siammask, cfg['hp'], device=device)
                state, bboxes, rboxes = siamese_track(value['state'], im, mask_enable=True,
                                                      refine_enable=True, device=device)
                value['state'] = state
                # Filter overlapping boxes
                if filter_boxes:
                    rboxes = filter_bboxes(rboxes, 10, c=10 * len(rboxes))

                if draw_candidates:
                    for box in range(len(rboxes)):
                        location = rboxes[box][0].flatten()
                        location = np.int0(location).reshape((-1, 1, 2))
                        traj = np.average(location, axis=0)[0]
                        frame_boxes.append([traj])
                        cv2.polylines(im, [location], True, colors[key - 1], 1)
                        cv2.circle(im, (int(traj[0]), int(traj[1])), 1, colors[key - 1], -1)

                location = value['state']['ploygon'].flatten()
                laloc = np.int0(location).reshape((-1, 1, 2))
                traj = np.int0(np.average(laloc, axis=0)[0])
                frame_boxes.append([traj])
                # all_centroids[key].append(frame_boxes)

                if bbox_type == 1:
                    # Work with axis-alligned bboxes
                    x1, y1, w1, h1 = get_aligned_bbox(location)
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h1)), colors[key - 1], 5)
                else:
                    cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, colors[key - 1], 3)

                # pred = append_pred(pred, f + 1, key, x1, y1, w1, h1)

                # Draw bounding box, centroid (and mask) of chosen candidate
                # cv2.circle(im, (int(traj[0]), int(traj[1])), 3, colors[key-1], -1)

                if draw_mask:
                    mask = state['mask'] > state['p'].seg_thr
                    im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

        cv2.imwrite(save_data_path + str(f).zfill(6) + '.jpeg', im)

        toc += cv2.getTickCount() - tic

    pred.to_csv(save_data_path + 'pred.txt', header=None, index=None, sep=',')

    with open(save_centroids_path, 'wb') as fil:
        pickle.dump(all_centroids, fil)

    toc /= cv2.getTickFrequency()
    fps = f / toc

    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps)'.format(toc, fps))
