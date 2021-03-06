import glob
import numpy as np
import cv2
import pickle
import argparse
import pandas as pd
import torch
from tools.test_multi import *
from custom import Custom
import glob
import pickle
# from tools.test import *
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from shapely.geometry import Polygon
from shapely.geometry import asPolygon

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', type=str, metavar='PATH',
                    default='SiamMask_DAVIS.pth', help='path to latest checkpoint')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
args = parser.parse_args()

# Definitions
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
columns_location = ['FrameID', 'ObjectID', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
dataset_name = ['MOT', 'SMOT', 'Stanford']
N = 1  # Maximum number of candidates returned by the tracking
k = 3  # Maximum number of candidates after filtering NMS
max_num_obj = 10  # Maximum number of objects being tracked

# Parameters
draw_mask = False
draw_candidates = False
filter_boxes = False
bbox_rotated = True
num_frames = 10  # 155 for Acrobats
dataset = 1  # 0: MOT, 1: SMOT, 2: Stanford
sequence = 'acrobats'
video = 'video0'

print('\nDataset:', dataset_name[dataset], ' Sequence:', sequence, ' Number of frames:', num_frames)

img_path, init_path, results_path, centroids_path, locations_path, dataset = get_paths(dataset, sequence, video)


def create_init(init_path, num_frames, max_num_obj):
    columns_standard = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive',
                        'isOccluded', 'cx', 'cy']
    init = pd.read_csv(init_path, sep=',', header=None)
    init.columns = columns_standard
    init = init[init.FrameID < num_frames]
    total_obj = init.ObjectID.unique()  # [5, 34, 65, 2...]
    if len(total_obj) > max_num_obj:
        total_obj = total_obj[:max_num_obj]
        init = init[init['ObjectID'].isin(total_obj)]
    return init, total_obj


# Loads init file, deletes targets if there are more than max_num_obj in the requested frames
init, total_obj = create_init(init_path, num_frames, max_num_obj)
pred_alig = init.copy()

print('Number of objects to be tracked:', len(total_obj))
colors = create_colormap_hsv(len(total_obj))

if __name__ == '__main__':
    toc = 0  # Timer
    # To compare performance with single object tracking
    pred_rot = pd.DataFrame(columns=columns_location)

    # Setup device and model
    torch.backends.cudnn.benchmark = True
    cfg = load_config(args)
    device = 2
    torch.cuda.set_device(device)
    # print('CUDA Available:', torch.cuda.device_count())
    print('CUDA device:', torch.cuda.current_device())


    # Initialize a SiamMask model for each object ID
    print('Initializing', len(total_obj), 'tracker(s)...')
    tracker = {}
    scores = {}
    for obj in total_obj:

        siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)
        siammask.eval().to(device)
        tracker[obj] = siammask
        scores[obj] = []
        print('Tracker:', obj, ' Initialized')

    # Parse Image files
    img_files = sorted(glob.glob(join(img_path, '*.jp*')))[000:num_frames]
    ims = [cv2.imread(imf) for imf in img_files]

    locations_dict = {}
    objects = {}
    with torch.no_grad():
        for f, im in enumerate(ims):
            print('------------------------ Frame:', f, '----------------------------')
            im_init = im.copy()
            im_track = im.copy()

            # cv2.putText(im, title, (10, 30), font, font_size, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(im, 'Frame: ' + str(f), (10, 60), font, font_size * 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            tic = cv2.getTickCount()

            # Get number of objects that are initialized in this frame
            init_frame = init[init.FrameID == f + 1]
            for index, row in init_frame.iterrows():
                ob = int(row['ObjectID'])
                x, y, w, h = row['x_topleft'], row['y_topleft'], row['Width'], row['Height']
                if ob in objects:
                    print('Object', ob, ' reinitialized')
                else:
                    locations_dict[ob] = []

                x1 = x + w
                y1 = y + h
                x2 = x
                y2 = y + h
                x3 = x
                y3 = y
                x4 = x + w
                y4 = y

                cx, cy = row['cx'], row['cy']
                locations_dict[ob].append([[np.array([x1, y1, x2, y2, x3, y3, x4, y4])]])

                nested_obj = {'target_pos': np.array([x + w / 2, y + h / 2]), 'target_sz': np.array([w, h]),
                              'init_frame': f, 'siammask': tracker[ob]}

                state, z = siamese_init(ob, im_init, nested_obj['target_pos'], nested_obj['target_sz'],
                                        nested_obj['siammask'], cfg['hp'], device=device)
                nested_obj['state'] = state
                objects[ob] = nested_obj
                scores[ob].append(1)
                cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 0), 5)
                cv2.putText(im, 'init', (int(x), int(y) - 7), font, font_size * 0.75, (255, 255, 0), 2, cv2.LINE_AA)

            for key, value in objects.items():
                col = colors[np.where(total_obj == key)[0][0]]
                print('Tracking object', key)
                centroids = []
                locations = []
                if value['init_frame'] == f:
                    print('Not going to track object', key, 'at this frame')
                    continue
                frame_boxes = []
                state, masks, rboxes_track = siamese_track_plus(state=value['state'], im=im_track, N=N,
                                                                mask_enable=True,
                                                                refine_enable=True, device=device)
                # state = siamese_track(state= value['state'], im=im_track, mask_enable=True,
                #                                           refine_enable=True, device=device)
                value['state'] = state
                scores[key].append(state['score'])

                if filter_boxes:  # Filter overlapping boxes
                    rboxes = filter_k_boxes(rboxes_track, k)
                else:
                    rboxes = rboxes_track


                for box in range(len(rboxes)):
                    location = np.int0(rboxes[box][0].flatten()).reshape((-1, 1, 2))
                    locations.append([rboxes[box][0].flatten()])
                    if draw_candidates:
                        cv2.polylines(im, [location], True, col, 1)

                locations_dict[key].append(locations)

                # TODO: Decide winner with Dynamics AND UPDATE TRACKER IF REQUIRED

                win = 0  # At this moment winner is the one that SiamMasks decides
                location = rboxes[win][0].flatten()
                pred_rot = append_pred_single(f, key, location, pred_rot)

                if bbox_rotated:
                    cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, col, 3)
                else:
                    # Work with axis-alligned bboxes
                    x1, y1, w1, h1 = get_aligned_bbox(location)
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h1)), col, 5)
                    pred_alig = append_pred(pred_alig, f + 1, key, x1, y1, w1, h1)

                if draw_mask:
                    for it, (box, score) in enumerate(rboxes_track):
                        if (box == rboxes[win][0]).all():
                            win_track = it
                            break
                    mask = masks[win_track] > state['p'].seg_thr
                    im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

            cv2.imwrite(results_path + str(f).zfill(6) + '.jpg', im)


    toc += cv2.getTickCount() - tic

    pred_alig.to_csv('/data/results/pred_alig_' + str(num_frames) + '.txt', header=None, index=None, sep=',')
    pred_rot.to_csv('/data/results/pred_rot_' + str(num_frames) + '.txt', header=None, index=None, sep=',')

    a = pred_rot.sum(axis=0, skipna=True).sum()
    # print('Difference when acrobats and num_frames=155:', int(3378487.46 - a))

    with open(locations_path, 'wb') as fil:
        pickle.dump(locations_dict, fil)

    with open('/data/Marina/centroids/scores.obj', 'wb') as fil:
        pickle.dump(scores, fil)


    toc /= cv2.getTickFrequency()
    fps = f / toc

    print('\nSiamMask Time: {:02.1f}s Speed: {:3.1f}fps)'.format(toc, fps))
    print('\nResults (frames with bboxes) saved in:', results_path)
    print('\n')
