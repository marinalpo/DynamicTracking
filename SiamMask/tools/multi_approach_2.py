import glob
import numpy as np
import cv2
import pickle
import argparse
import pandas as pd
import torch
from tools.test_multi import *
from custom import Custom
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from dynamics.Tracker_Dynamics_2 import TrackerDyn_2
from utils_dyn.utils_plots_dynamics import *
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# Parsing
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
columns_names = ['FrameID',	'ObjectID',	'x_topleft',	'y_topleft',	'Width',	'Height',	'isActive',	'isOccluded', 'cx', 'cy']
max_num_obj = 10  # Maximum number of objects being tracked

# Tracker Parameters
correct_with_dynamics = False
draw_GT = True
draw_proposal = True
draw_candidates = True
draw_pred = True

draw_mask = False
filter_boxes = True
eps = 1  # Noise variance
metric = 0  # if 0: JBLD, if 1: JKL
W = 3  # Smoothing window length
slow = False  # If true: Slow(but Precise), if false: Fast
norm = True  # If true: Norm, if false: MSE

T0 = 11  # System memory (best: 11)
num_frames = 150  # 150 for Acrobats
N = 30  # Maximum number of candidates returned by the tracking (15)
k = 3  # Maximum number of candidates
# after filtering NMS
dataset = 1  # 0: MOT, 1: SMOT, 2: Stanford
sequence = 'acrobats'  # SMOT: 'acrobats' or 'juggling'
video = 'video0'

print('\nDataset:', dataset_name[dataset], ' Sequence:', sequence, ' Number of frames:', num_frames)

img_path, init_path, results_path, centroids_path, locations_path, dataset, gt_path = get_paths(dataset, sequence, video)


# TODO: Si el volem nomes dun objecte
single_object = True
obj = 5

# Loads init file, deletes targets if there are more than max_num_obj in the requested frames
init, total_obj = create_init(init_path, num_frames, max_num_obj)
pred_df = init.copy()
gt_df = pd.read_csv(gt_path, sep=',', header=None)
gt_df.columns = columns_names
gt_df = gt_df[gt_df.FrameID <= num_frames]

print('\ntotal obj:', total_obj, '\n')


colors = create_colormap_hsv(len(total_obj))

if single_object:
    total_obj = [obj]
    init = init[init.ObjectID == obj]
    gt_df = gt_df[gt_df.ObjectID == obj]
    c_gt = gt_df.loc[:, ['cx', 'cy']].values
    gt_df = gt_df[gt_df.ObjectID == obj]

print('init:\n', init)

if __name__ == '__main__':
    toc = 0  # Timer
    # To compare performance with single object tracking
    pred_rot = pd.DataFrame(columns=columns_location)

    # Setup device and model
    torch.backends.cudnn.benchmark = True
    cfg = load_config(args)
    device = 2
    torch.cuda.set_device(device)
    # print('CUDA device:', torch.cuda.current_device())

    # Initialize a SiamMask model for each object ID
    # print('Initializing', len(total_obj), 'tracker(s)...')
    tracker = {}
    dynamics = {}
    for obj in total_obj:
        # Initialize tracker, and DynTracking for each object
        siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)
        siammask.eval().to(device)
        tracker[obj] = siammask

        # print('Tracker:', obj, ' Initialized')

    # Parse Image files
    img_files = sorted(glob.glob(join(img_path, '*.jp*')))[000000:num_frames]
    ims = [cv2.imread(imf) for imf in img_files]

    locations_dict = {}
    target_sz_dict = {}
    target_pos_dict = {}
    objects = {}

    with torch.no_grad():
        for f, im in enumerate(ims):
            f = f + 1  # Begins with image 000001.jpg
            print('------------------------ Frame:', f, '----------------------------')
            im_init = im.copy()
            im_track = im.copy()
            tic = cv2.getTickCount()
            # cv2.putText(im, 'Frame: ' + str(f), (10, 60), font, font_size * 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # Get number of objects that are initialized in this frame
            init_frame = init[init.FrameID == f]
            for index, row in init_frame.iterrows():
                ob = int(row['ObjectID'])
                x, y, w, h = row['x_topleft'], row['y_topleft'], row['Width'], row['Height']

                if ob in objects:  # IF reinizialization
                    print('Object', ob, ' reinitialized')

                else:
                    locations_dict[ob] = []
                    target_sz_dict[ob] = []
                    target_pos_dict[ob] = []

                # x1, y1, x2, y2, x3, y3, x4, y4 = x + w, y + h, x, y + h, x, y, x + w, y
                cx, cy = row['cx'], row['cy']
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])

                # locations_dict[ob].append([[np.array([x1, y1, x2, y2, x3, y3, x4, y4])]])
                target_sz_dict[ob].append(np.array(target_sz))
                target_pos_dict[ob].append(np.array(target_pos))

                torch.cuda.set_device(device)


                tracker_dyn = TrackerDyn_2(T0=T0, t_init=f)
                c, pred_pos, pred_ratio = tracker_dyn.update(target_pos, target_sz, 1)

                nested_obj = {'target_pos': target_pos, 'target_sz': np.array([w, h]),
                              'init_frame': f, 'siammask': tracker[ob], 'tracker': tracker_dyn}

                state, z = siamese_init(ob, im_init, nested_obj['target_pos'], nested_obj['target_sz'],
                                        nested_obj['siammask'], cfg['hp'], device=device)

                nested_obj['state'] = state
                objects[ob] = nested_obj
                cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 0), 5)
                cv2.putText(im, 'init obj:'+str(ob), (int(x), int(y) - 7), font, font_size * 0.75, (255, 255, 0), 2, cv2.LINE_AA)

            for key, value in objects.items():
                if value['init_frame'] == f:
                    print('Not going to track object', key, 'at this frame')
                    continue

                print('Tracking object', key)
                state = value['state']
                tracker_dyn = value['tracker']

                # col = colors[np.where(total_obj == key)[0][0]]
                col = colors[key-1]

                state, masks, rboxes_cand, bboxes = siamese_track_plus(state=value['state'], im=im_track, N=N,
                                                                mask_enable=True,
                                                                refine_enable=True, device=device)

                target_sz = state['target_sz']
                target_pos = state['target_pos']
                score = state['score']
                poly = state['ploygon']

                if draw_proposal:
                    cx = target_pos[0]
                    cy = target_pos[1]
                    w = target_sz[0]
                    h = target_sz[1]
                    x_tl = cx - w / 2
                    y_tl = cy - h / 2
                    cv2.rectangle(im, (int(x_tl), int(y_tl)), (int(x_tl + w), int(y_tl + h)), (255, 255, 255), 3)

                if draw_candidates:
                    num_cand = bboxes.shape[1]
                    for i in range(num_cand):
                        cx, cy = bboxes[0, i], bboxes[1, i]
                        w, h = bboxes[2, i], bboxes[3, i]
                        # cv2.circle(im, (int(cx), int(cy)), 3, (0, 254, 0), 3)
                        cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (254, 254, 254), 1)


                bbox_ratio = target_pos[0]/target_pos[1]
                # print('bbox ratio:', bbox_ratio)

                if correct_with_dynamics:
                    c, pred_pos, pred_ratio = tracker_dyn.update(target_pos, target_sz, score)

                    if c[0] or c[1]:
                        # Remove winner bbox
                        bboxes = bboxes[:, 1:]

                        pred_pos, pred_sz, idx = get_best_bbox(bboxes, pred_pos, pred_ratio)
                        tracker_dyn.update_bbox(pred_pos, pred_sz)

                        state['target_pos'] = pred_pos
                        state['target_sz'] = pred_sz


                value['state'] = state

                cx = state['target_pos'][0]
                cy = state['target_pos'][1]
                w = state['target_sz'][0]
                h = state['target_sz'][1]
                x_tl = cx - w/2
                y_tl = cy - h/2

                # Append predictions to pred DF
                # x_topleft, y_topleft, w, h, cx, cy
                pred_df = append_pred(pred_df, f, key, x_tl, y_tl, w, h, cx, cy)

                if draw_pred:
                    cv2.rectangle(im, (int(x_tl), int(y_tl)), (int(x_tl + w), int(y_tl + h)), col, 3)
                    cv2.circle(im, (int(x_tl), int(y_tl)), 3, col, 3)

                if draw_GT:
                    line = gt_df[(gt_df.FrameID == f) & (gt_df.ObjectID == key)]
                    gt_sz = np.asarray((line.Width.item(), line.Height.item()))
                    gt_pos = np.asarray((line.cx.item(), line.cy.item()))
                    cx_gt = gt_pos[0]
                    cy_gt = gt_pos[1]
                    w_gt = gt_sz[0]
                    h_gt = gt_sz[1]
                    x_tl_gt = cx_gt - w_gt / 2
                    y_tl_gt = cy_gt - h_gt / 2
                    cv2.rectangle(im, (int(x_tl_gt), int(y_tl_gt)), (int(x_tl_gt + w_gt), int(y_tl_gt + h_gt)), col, 1)
                    cv2.circle(im, (int(x_tl_gt), int(y_tl_gt)), 3, col, 1)


            cv2.imwrite(results_path + str(f).zfill(6) + '.jpg', im)
            toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    fps = f / toc

    # print('pred_df:\n', pred_df)
    pred_df.to_csv('/data/results/pred_df_acrobats.txt', header=None, index=None, sep=',')
    gt_df.to_csv('/data/results/gt_df_acrobats.txt', header=None, index=None, sep=',')

    positions_path = '/data/Marina/positions/'

    with open(locations_path, 'wb') as fil:
        pickle.dump(locations_dict, fil)

    with open(positions_path+'target_sz_dict.obj', 'wb') as fil:
        pickle.dump(target_sz_dict, fil)

    with open(positions_path+'target_pos_dict.obj', 'wb') as fil:
        pickle.dump(target_pos_dict, fil)



    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps)'.format(toc, fps))

    # TODO: Metrics computation
    mED, mIOU, mP, mota, motp = compute_metrics(gt_df, pred_df, th=0.8)

    print('\n------------------- MEASURES REPORT: -------------------')
    print('mED:', mED, 'pixels')
    print('mIOU:', mIOU, '%')
    print('mP(@0.8):', mP, '%')
    print('MOTA(@0.8):', mota, '%')
    print('MOTP(@0.8):', motp, '%')
    print('--------------------------------------------------------\n')
    #
    #
    if single_object and correct_with_dynamics:
        tin = tracker_dyn.t_init
        tfin = num_frames
        c_gt = c_gt[tin-1:tfin+1, :]
        print('tin', tin, 'tfin', tfin)
        plot_jbld_eta_score_4(tracker_dyn, c_gt, obj, norm, slow, tin, tfin)
        plot_gt_cand_pred_box(tracker_dyn, c_gt, obj, norm, slow, tin, tfin)
