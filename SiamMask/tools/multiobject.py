import glob
import numpy as np
import cv2
import pickle
import argparse
import pandas as pd
import torch
import json
from tools.test_1_nostre import *
from utils.bbox_helper import get_aligned_bbox
from shapely.geometry import asPolygon
from custom import Custom

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', type=str, metavar='PATH',
                    default='SiamMask_DAVIS.pth', help='path to latest checkpoint')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
args = parser.parse_args()

# Definitions
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
columns_standard = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'isOccluded']
dataset_name = ['MOT', 'SMOT', 'Stanford']

# Parameters
draw_mask = True
draw_candidates = False
filter_boxes = False
bbox_rotated = True
num_frames = 40
dataset = 1  # 0: MOT, 1: SMOT, 2: Stanford
sequence = 'acrobats'
video = 'video0'
max_num_obj = 2
print('\nDataset:', dataset_name[dataset], ' Sequence:', sequence)


img_path, init_path, results_path, centroids_path, dataset = get_paths(dataset, sequence, video)
init_path = '/data/SMOT/' + sequence + '/gt/init_old.txt'
title = 'Dataset: ' + dataset + ' Sequence: ' + sequence
init = pd.read_csv(init_path, sep=',', header=None)
print(init)

# Assign column names
init.columns = columns_standard
init = init[init.FrameID < num_frames]
# TODO: Get the only 10 objects that appear
pred = init.copy()

total_obj = init.ObjectID.unique()  # [5, 34, 65, 2...]
print('total object', total_obj)


colors = create_colormap_hsv(5)

if __name__ == '__main__':
    toc = 0  # Timer

    # Setup device and model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = load_config(args)

    # Initialize a SiamMask model for each object ID
    print('Initializing', len(total_obj), 'tracker(s)...')

    tracker = {}
    for obj in total_obj:
        siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)
        siammask.eval().to(device)
        tracker[obj] = siammask
        print('Tracker:', obj, ' Initialized')


    # Parse Image files
    img_files = sorted(glob.glob(join(img_path, '*.jp*')))[000:num_frames]
    ims = [cv2.imread(imf) for imf in img_files]
    im_ori = ims[0]  # First frame

    all_centroids = {}  # Dict that will save all the objects trajectories and discarded candidates
    objects = {}

    for f, im in enumerate(ims):
        print('Frame:', f + 1)
        cv2.putText(im, title, (10, 30), font, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, 'Frame: ' + str(f), (10, 60), font, font_size * 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tic = cv2.getTickCount()

        # Get number of objects that are initialized in this frame
        init_frame = init[init.FrameID == f + 1]
        for index, row in init_frame.iterrows():
            ob = int(row['ObjectID'])
            print('OB', ob)
            x = row['x_topleft']
            y = row['y_topleft']
            w = row['Width']
            h = row['Height']

            # if ob in objects:
            #     print('OBJECT', ob, ' IS ALREADY IN THE DICTIONARY')
            #     siammask = Custom(anchors=cfg['anchors'])
            #     if args.resume:
            #         assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            #         siammask = load_pretrain(siammask, args.resume)
            #     siammask.eval().to(device)
            #     list_siammasks[ob - 1] = siammask

            nested_obj = {'target_pos': np.array([x + w / 2, y + h / 2]), 'target_sz': np.array([w, h]),
                              'init_frame': f, 'siammask': tracker[ob]}
            
            state = siamese_init(im, nested_obj['target_pos'], nested_obj['target_sz'],  nested_obj['siammask'],
                                 cfg['hp'], device=device)
            nested_obj['state'] = state
            objects[ob] = nested_obj
            cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), colors[ob - 1], 5)

        if f > 0:  # Perform tracking
            for key, value in objects.items():
                print('Key:', key)
                if value['init_frame'] == f:
                    print('Not going to track at this frame')
                    continue
                frame_boxes = []
                # state, bboxes, rboxes = siamese_track(value['state'], im, value['siammask'], mask_enable=True,
                #                                       refine_enable=True, device=device)
                state = siamese_track(key, f, value['state'], im, mask_enable=True,
                                                      refine_enable=True, device=device)
                value['state'] = state
                # # Filter overlapping boxes
                # if filter_boxes:
                #     rboxes = filter_bboxes(rboxes, 10, c=10 * len(rboxes))
                #
                # if draw_candidates:
                #     for box in range(len(rboxes)):
                #         location = rboxes[box][0].flatten()
                #         location = np.int0(location).reshape((-1, 1, 2))
                #         traj = np.average(location, axis=0)[0]
                #         frame_boxes.append([traj])
                #         cv2.polylines(im, [location], True, colors[key - 1], 1)
                #         # Draw Centroids
                #         # cv2.circle(im, (int(traj[0]), int(traj[1])), 1, colors[key - 1], -1)
                #
                # location = value['state']['ploygon'].flatten()
                # laloc = np.int0(location).reshape((-1, 1, 2))
                # traj = np.int0(np.average(laloc, axis=0)[0])
                # frame_boxes.append([traj])
                # # all_centroids[key].append(frame_boxes)
                #
                # if bbox_rotated:
                #     cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, colors[key - 1], 3)
                # else:
                #     # Work with axis-alligned bboxes
                #     x1, y1, w1, h1 = get_aligned_bbox(location)
                #     cv2.rectangle(im, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h1)), colors[key - 1], 5)
                #
                #
                # # pred = append_pred(pred, f + 1, key, x1, y1, w1, h1)
                #
                # # Draw bounding box, centroid (and mask) of chosen candidate
                # # cv2.circle(im, (int(traj[0]), int(traj[1])), 3, colors[key-1], -1)
                #
                # if draw_mask:
                #     mask = state['mask'] > state['p'].seg_thr
                #     im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

                # location = state['ploygon'].flatten()
                # mask = state['mask'] > state['p'].seg_thr
                # im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                # laloc = np.int0(location).reshape((-1, 1, 2))
                # traj = np.int0(np.average(laloc, axis=0)[0])
                # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        # cv2.imwrite(results_path + str(f).zfill(6) + '.jpg', im)

        toc += cv2.getTickCount() - tic

    pred.to_csv(results_path + 'pred.txt', header=None, index=None, sep=',')

    with open(centroids_path, 'wb') as fil:
        pickle.dump(all_centroids, fil)

    toc /= cv2.getTickFrequency()
    fps = f / toc

    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps)'.format(toc, fps))
