import glob
import numpy as np
import cv2
import pickle
import argparse
import json
from tools.test_2_nostre import *
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from shapely.geometry import asPolygon
from custom import Custom

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

# Video Parameters
drawMask = True
drawCandidates = True
filterBoxes = False
num_frames = 100
title = 'Dataset: VOT2018 ' \
        'Sequence: ants1'
save_data_path = '/data/Marina/ants1/results/'
base_path = '/data/Marina/ants1/images/'
save_centroids_path = '/data/Marina/ants1/points/centroids_ant1.obj'
init_rects = [(107, 445, 62, 100)]
# init_rects = [(107, 445, 62, 100), (236, 211, 104, 67), (187, 621, 88, 60),
#               (225, 561, 90, 67), (166, 444, 83, 58), (233, 455, 60, 93)]


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

    n_obj = len(init_rects)  # Number of objects being tracked
    target_pos = np.zeros((n_obj, 2))  # Targets centroids in the first frame
    target_sz = np.zeros((n_obj, 2))  # Targets height and width
    for i in range(n_obj):
        x, y, w, h = init_rects[i]
        target_pos[i, :] = np.array([x + w / 2, y + h / 2])
        target_sz[i, :] = np.array([w, h])

    all_centroids = {}  # Dict that will save all the objects trajectories and discarded candidates

    for f, im in enumerate(ims):
        print('Frame:', f)
        cv2.putText(im, title, (10, 30), font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im, 'Frame: ' + str(f), (10, 60), font, font_size*0.75, (0, 0, 0), 2, cv2.LINE_AA)
        tic = cv2.getTickCount()

        if f == 0:  # Initialize tracker
            list_states = []
            for i in range(n_obj):
                dict_key = 'object' + str(i)  # object1, object2, ...
                all_centroids[dict_key] = []
                state = siamese_init(im_ori, target_pos[i, :], target_sz[i, :], siammask, cfg['hp'], device=device)  # init tracker
                list_states.append(state)

        elif f > 0:  # Perform tracking
            for i in range(n_obj):
                dict_key = 'object' + str(i)  # object1, object2, ...
                frame_boxes = []
                state_obj = list_states[i]
                state_init = siamese_init(im_ori, target_pos[i, :], target_sz[i, :], siammask, cfg['hp'], device=device)
                state, bboxes, rboxes = siamese_track(state_obj, im, mask_enable=True, refine_enable=True, device=device)
                list_states[i] = state

                # Filter overlapping boxes
                if filterBoxes:
                    rboxes = filter_bboxes(rboxes, 10, c=10 * len(rboxes))

                if drawCandidates:
                    for box in range(len(rboxes)):
                        location = rboxes[box][0].flatten()
                        location = np.int0(location).reshape((-1, 1, 2))
                        traj = np.average(location, axis=0)[0]
                        frame_boxes.append([traj])
                        cv2.polylines(im, [location], True, colors[i], 1)
                        cv2.circle(im, (int(traj[0]), int(traj[1])), 1, colors[i], -1)

                location = state['ploygon'].flatten()
                laloc = np.int0(location).reshape((-1, 1, 2))
                traj = np.int0(np.average(laloc, axis=0)[0])
                frame_boxes.append([traj])
                all_centroids[dict_key].append(frame_boxes)

                # Draw bounding box, centroid (and mask) of chosen candidate
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, colors[i], 3)
                cv2.circle(im, (int(traj[0]), int(traj[1])), 3, colors[i], -1)
                if drawMask:
                    mask = state['mask'] > state['p'].seg_thr
                    im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

            cv2.imwrite(save_data_path + str(f) + '.jpeg', im)

        toc += cv2.getTickCount() - tic

    with open(save_centroids_path, 'wb') as fil:
        pickle.dump(all_centroids, fil)

    toc /= cv2.getTickFrequency()
    fps = f / toc

    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps)'.format(toc, fps))
