# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import math
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile
import pandas as pd

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str
from shapely.geometry import asPolygon

thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom', ],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')




def append_pred_single(f, ob, location, df):
  columns_location = ['FrameID', 'ObjectID', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
  location = location.tolist()
  location.insert(0, f)
  location.insert(1, ob)
  df = df.append(dict(zip(df.columns, location)), ignore_index=True)
  return df


def get_paths(dataset, sequence, video='video0'):

    if dataset == 0:
        dataset = '2DMOT2015'
        root = '/data/2DMOT2015/train/'
        img_path = root + sequence + '/img1/'
        init_path = root + sequence + '/gt/init.txt'

    elif dataset == 1:  # SMOT
        dataset = 'SMOT'
        img_path = '/data/SMOT/' + sequence + '/img/'
        init_path = '/data/SMOT/' + sequence + '/gt/init.txt'

    elif dataset == 2:  # Stanford
        dataset = 'stanford-campus'
        img_path = '/data/stanford-campus/videos/' + video + '/'
        init_path = '/data/stanford-campus/annotations/' + sequence + '/' + video + '/init.txt'

    results_path = '/data/results/'
    centroids_path = '/data/Marina/centroids/centroids_' + dataset + '_' + sequence + '.obj'
    locations_path = '/data/Marina/centroids/locations_' + dataset + '_' + sequence + '.obj'

    return img_path, init_path, results_path, centroids_path, locations_path, dataset


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


arrendatario = 0


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    # im: image
    # pos: target position (centroid x, y)
    # model_sz: static (127 init, 255 tracking)
    # original_sz: search area (216... depen de la mida del target)

    # print('PRINTS FROM get_subwindow_tracking:')

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    # print('context_xmin, context_xmax, context_ymin, context_ymax, left_pad, top_pad, right_pad, bottom_pad')
    # print(context_xmin, context_xmax, context_ymin, context_ymax, left_pad, top_pad, right_pad, bottom_pad)
    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        # print('Entra if 1')
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        # print('NO entra if 1')
        # print('context_xmax, context_xmin, context_ymax, context_ymin:')
        # print(context_xmax, context_xmin, context_ymax, context_ymin)
        # print('im sum:')
        # print(im.sum())
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        # print('Entra if 2')
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        # print('NO entra if 2')
        im_patch = im_patch_original
    # np.save('/data/Marina/patch.npy', im_patch)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch



def compute_intersection_by_pairs(polygons, sorted_scores, intersection_th=0.5):
    "Computes "
    results = []
    for i in range(polygons.shape[0] // 2):
        pol_a = asPolygon(polygons[2*i, :, :])
        pol_b = asPolygon(polygons[2*i + 1, :, :])
        intersection_area = pol_a.intersection(pol_b)
        iou = intersection_area.area / (pol_a.area + pol_b.area)
        if iou >= intersection_th:
            if (sorted_scores[2*i] >= sorted_scores[2*i + 1]):
                results.append([polygons[2*i, :, :], sorted_scores[2*i]])
            else:
                results.append([polygons[2*i + 1, :, :], sorted_scores[2*i+1]])
        else:
            results.append([polygons[2*i, :, :], sorted_scores[2*i]])
            results.append([polygons[2*i + 1, :, :], sorted_scores[2*i+1]])
    return results



def filter_bboxes(rboxes, k):
    """
    rboxes: list, contains: [(np.array(polygon), score),(), ... , ()]
    """

    num_bxs = len(rboxes)

    c = 5*len(rboxes)
    if num_bxs <= 1:
        return rboxes
    if num_bxs == k:
        return rboxes

    polygons = np.zeros((num_bxs, 4, 2))  # (num_bxs, 8 (points of a polygon, 2 (coordinates of each point)))
    scores = np.zeros(num_bxs)
    for i, (box, score) in enumerate(rboxes):
        box_mat = box.reshape(4, 2)
        polygons[i, :, :] = box_mat
        scores[i] = score
    # Compute centroids
    centroids = np.mean(polygons, axis=1)
    indexes_x = np.argsort(centroids[:, 0])  # Ascending order
    indexes_y = np.argsort(centroids[:, 1])
    centroids_x = centroids[indexes_x, 0]
    centroids_y = centroids[indexes_y, 1]

    xmax, xmin = centroids_x[-1], centroids_x[0]
    ymax, ymin = centroids_y[-1], centroids_y[0]
    intersection_th = 0.25  # If 2 boxes overlap more than 0.4 the one with max score will remain

    if ((xmax - xmin) >= (ymax - ymin)):
        # Make pairs according to x axis (indexes_x)
        polygons = polygons[indexes_x]
        sorted_scores = scores[indexes_x]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores, intersection_th)
    else:
        # Make pairs according to y axis (indexes_y)
        polygons = polygons[indexes_y]
        sorted_scores = scores[indexes_y]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores, intersection_th)
    if (len(rboxes) != 1 and len(rboxes) > k):
        c -= 1
        rboxes = filter_bboxes(rboxes, k)

    return rboxes


def filter_k_boxes(rboxes, k):
    winner = rboxes[0]
    rboxes_filt = filter_bboxes(rboxes, k)
    for i, (box, score) in enumerate(rboxes_filt):
        if (box == winner[0]).all():
            del rboxes_filt[i]
    rboxes_filt.insert(0, list(winner))
    return rboxes_filt


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors

    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(ob, im, target_pos, target_sz, model, hp=None, device='cpu'):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    p = TrackerConfig()  # default hyper-params for SiamMask: stride, learning rate...
    p.update(hp, model.anchors)
    # hp: 'instance_size': 255, 'base_size': 8, 'out_size': 127, 'seg_thr': 0.35,
    # 'penalty_k': 0.04, 'window_influence': 0.4, 'lr': 1.0
    # model.anchors: {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'round_dight': 0}
    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))  # Search area

    # initialize the exemplar
    # print('PRINTS FROM init:')

    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    # z_crop size: torchSize([3, 127, 127)]
    # print('Sum z crop')
    # print(z_crop.sum())

    z = Variable(z_crop.unsqueeze(0))

    # La xarxa es guarda les features resultants (self.zf) d'haver passat el patch z per la siamesa
    net.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    print("window = ", state['window'])
    return state, z_crop


def siamese_track_plus(state, im, N, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    global arrendatario
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x  # p.exemplar_size = 127, sempre es la mateixa
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    debug = False
    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 255, 0), 2)
        # cv2.imwrite('/data/Ponc/tracking/results/windows-seagulls-debug/'+'search_'+str(arrendatario)+'.jpeg', im_debug)
        cv2.waitKey(0)

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # In davis we have 5 anchors
    if mask_enable:
        score, delta, mask = net.track_mask(
            x_crop.to(device))  # score: (1,10,25,25), delta: (1, 20 (5boxes*4coords), 25, 25), mask: (1, 63*63, 25, 25)
    else:
        score, delta = net.track(x_crop.to(device))

    # np.save('/data/Marina/x_crop2.npy', x_crop.data.cpu().numpy())
    # np.save('/data/Marina/delta.npy', delta.data.cpu().numpy())
    # np.save('/data/Marina/score.npy', score.data.cpu().numpy())

    # delta: torch.Tensor shape: ([1, 20, 25, 25])
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    # np.save('/data/Marina/delta2.npy', delta)
    # delta: numpy.ndarray shape: (4, 3125)

    # Softmax in 3125,2,which each column is BG, FG
    # Et quedes amb data[1] (el foreground): (3125, )
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    np.save('/data/Marina/delta2.npy', delta)

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz * scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    # penalitza si esta mes lluny del centre
    pscore = penalty * score

    # NOTE: Comença la nostre aportació

    # cos window (motion model)
    # N = 17  # TODO: Passar d'argument
    bboxes = np.zeros((6, N), dtype=np.float64)
    # bboxes has the shape (6 , Npoints) ; 0=res_x, 1=res_y, 2=res_w, 3=res_h, 4=score, 5=best_pscore_id_tmp
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    attmap = score.reshape(5, 25, 25)
    attmap = np.amax(attmap, axis=0)
    np.save('/data/Marina/attmap.npy', attmap)

    best_score_threshold = 0.99
    for idx in range(0, N):
        if idx == 0:
            best_pscore_id = np.argmax(pscore)

        best_pscore_id_tmp = np.argmax(pscore)
        pred_in_crop = delta[:, best_pscore_id_tmp] / scale_x

        lr = penalty[best_pscore_id_tmp] * score[best_pscore_id_tmp] * p.lr  # lr for OTB

        res_x = pred_in_crop[0] + target_pos[0]
        res_y = pred_in_crop[1] + target_pos[1]
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        bboxes[0, idx] = target_pos[0]
        bboxes[1, idx] = target_pos[1]
        bboxes[2, idx] = target_sz[0]
        bboxes[3, idx] = target_sz[1]
        bboxes[4, idx] = score[best_pscore_id_tmp]  # BUG: This should be pscore[best_...]?
        bboxes[5, idx] = best_pscore_id_tmp

        # if (pscore[best_pscore_id] > best_score_threshold):
        #     break
        pscore[best_pscore_id_tmp] = 0.0

    # Tot de la millor pscore (la guanyadora original)
    target_pos = np.array([bboxes[0, 0], bboxes[1, 0]])
    target_sz = np.array([bboxes[2, 0], bboxes[3, 0]])
    # for Mask Branch
    rboxes = []
    deltas = []
    list_masks = []
    for idx in range(0, N):
        if mask_enable:
            best_pscore_id_mask = np.unravel_index(int(bboxes[5, idx]), (5, p.score_size, p.score_size))
            delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]
            # delta_x and delta_y are the selected coordinates in the volume

            # NOTE: Nomes agafo boxes de deltes noves, no vull coses de la mateixa delta
            if ((delta_x, delta_y) not in deltas):
                # print("delta: (", delta_x, ", ", delta_y, ")")
                deltas.append((delta_x, delta_y))
                if refine_enable:
                    mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                        p.out_size, p.out_size).cpu().data.numpy()
                else:
                    mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                        squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

                def crop_back(image, bbox, out_sz, padding=-1):
                    a = (out_sz[0] - 1) / bbox[2]
                    b = (out_sz[1] - 1) / bbox[3]
                    c = -a * bbox[0]
                    d = -b * bbox[1]
                    mapping = np.array([[a, 0, c],
                                        [0, b, d]]).astype(np.float)
                    crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=padding)
                    return crop

                s = crop_box[2] / p.instance_size
                sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                           crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                           s * p.exemplar_size, s * p.exemplar_size]
                s = p.out_size / sub_box[2]
                back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
                mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))
                list_masks.append(mask_in_img)

                target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
                if cv2.__version__[-5] == '4':
                    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                else:
                    _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                if len(contours) != 0 and np.max(cnt_area) > 100:
                    contour = contours[np.argmax(cnt_area)]  # use max area polygon
                    polygon = contour.reshape(-1, 2)
                    # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
                    prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

                    # box_in_img = pbox
                    rbox_in_img = prbox
                    box_score = bboxes[4, idx]
                    rboxes.append([rbox_in_img, box_score])
                    # if(len(deltas) == 1):
                    #     attmap[delta_x, dhe elta_y] = 3.5
                    # else:
                    #     attmap[delta_x, delta_y] = 1.5

                    if debug:

                        im_debug_overlay = im_debug.copy()
                        im_debug_overlay[:, :, :] = np.array([0.0, 0.0, 0.0])
                        torch_data = np.float64(im_debug_overlay[:, :, 0].copy())
                        patch_size = crop_box_int[0]
                        num_deltas = 25  # aixo no varia
                        patch_ratio = int(patch_size / num_deltas)  # aixo varia
                        resized_img_h, resized_img_w = im_debug.shape[0] / 5, im_debug.shape[1] / 5

                        torch_data_delta_size = np.zeros((int(resized_img_h), int(resized_img_w)))
                        offset_x_deltas = int(crop_box_int[0] / 5)
                        offset_y_deltas = int(crop_box_int[1] / 5)

                        length_x = int((crop_box_int[2]) / 5)
                        length_y = int((crop_box_int[3]) / 5)

                        for i in range(25):
                            for j in range(25):
                                # step_x = crop_box_int[0] + i*length_x
                                # step_y = crop_box_int[1] + j*length_y
                                # im_debug_overlay[step_y: step_y + length_y, step_x: step_x+length_x, :] = np.array([0.0,0.0,0.0])
                                # im_debug_overlay[step_y: step_y + length_y, step_x: step_x+length_x, :] = np.uint8(attmap[j,i] * np.array([0,165,255]))
                                # torch_data[step_y: step_y + length_y, step_x: step_x+length_x] = attmap[j,i]*1.0
                                # Now for the reshaped
                                torch_data_delta_size[offset_y_deltas + j, offset_x_deltas + i] = attmap[j, i] * 1.0
                                if (pscore[best_pscore_id] > best_score_threshold):
                                    torch_data_delta_size[offset_y_deltas + delta_y, offset_x_deltas + delta_x] = 3.0

                        if (pscore[best_pscore_id] > best_score_threshold):
                            sma = torch.nn.Softmax()
                            torch_data_delta_size = sma(torch.from_numpy(np.exp(torch_data_delta_size))).numpy()

                            # im_debug_overlay[step_x: step_x+length_x, step_y: step_y + length_y, :] = attmap[j,i]

                        overlay_result = cv2.addWeighted(im_debug, 0.70, im_debug_overlay, 0.3, 0.0)
                        # cv2.imwrite('/data/Ponc/tracking/results/windows-seagulls-debug/'+'search_'+str(arrendatario)+'.jpeg', overlay_result)
                        # np.save('/data/Ponc/tracking/torch_data/resized/'+"{:05d}".format(arrendatario)+'.npy', torch_data_delta_size)

                    # np.save('/data/Ponc/tracking/results/mevasa/'+"{:05d}".format(arrendatario)+'.npy', attmap)
                else:  # empty mask
                    location = cxy_wh_2_rect(target_pos, target_sz)
                    rbox_in_img = np.array([[location[0], location[1]],
                                            [location[0] + location[2], location[1]],
                                            [location[0] + location[2], location[1] + location[3]],
                                            [location[0], location[1] + location[3]]])
        # if (pscore[best_pscore_id] > best_score_threshold):
        #     break

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = bboxes[4, 0]
    state['mask'] = list_masks[0]
    state['ploygon'] = rboxes[0][0]
    return state, list_masks, rboxes


def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    global arrendatario
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x  # p.exemplar_size = 127, sempre es la mateixa
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    debug = False
    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 255, 0), 2)
        cv2.putText(im_debug, 'Frame:'+str(f), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im_debug, 'key:' + str(key), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite('/data/results/search_' +str(f)+'_key_'+str(key)+'single.jpeg', im_debug)
        # cv2.waitKey(0)

    # extract scaled crops for search region x at previous target position
    # print('sx track abans dentrar getsubwindowtracking:', s_x.sum())
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    # print('Sum x crop')
    # print(x_crop.sum())
    # In davis we have 5 anchors
    if mask_enable:
        score, delta, mask = net.track_mask(
            x_crop.to(device))  # score: (1,10,25,25), delta: (1, 20 (5boxes*4coords), 25, 25), mask: (1, 63*63, 25, 25)
    else:
        score, delta = net.track(x_crop.to(device))

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()

    # Softmax in 3125,2,which each column is BG, FG
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz * scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)

    # bboxes has the shape (6 , Npoints) ; 0=res_x, 1=res_y, 2=res_w, 3=res_h, 4=score, 5=best_pscore_id_tmp
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence

    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x

    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]
    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch

    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[
            1]  # delta_x and delta_y are the selected coordinates in the volume
        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state


def track_vot(model, video, hp=None, mask_enable=False, refine_enable=False, device='cpu'):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, model, hp, device)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            state = siamese_track(state, im, mask_enable, refine_enable, device, args.debug)  # track
            if mask_enable:
                location = state['ploygon'].flatten()
                mask = state['mask']
            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                mask = []

            if 'VOT' in args.dataset:
                gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                              (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
                if mask_enable:
                    pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
                else:
                    pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
                b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            else:
                b_overlap = 1

            if b_overlap:
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]),
                                  (0, 255, 0), 3)
            if len(location) == 8:
                if mask_enable:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im_show, str(state['score']) if 'score' in state else '', (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    name = args.arch.split('.')[0] + '_' + ('mask_' if mask_enable else '') + ('refine_' if refine_enable else '') + \
           args.resume.split('/')[-1].split('.')[0]

    if 'VOT' in args.dataset:
        video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                    fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    else:  # OTB
        video_path = join('test', args.dataset, name)
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x]) + '\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    output_max_id = np.argmax(outputs, axis=0).astype('uint8') + 1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j + 1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res


def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, device='cpu'):
    image_files = video['image_files']

    annos = [np.array(Image.open(x)) for x in video['anno_files']]
    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init * len(object_ids)
    object_num = len(object_ids)
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1])) - 1
    for obj_id, o_id in enumerate(object_ids):

        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            tic = cv2.getTickCount()
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id
                x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                cx, cy = x + w / 2, y + h / 2
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                state = siamese_init(im, target_pos, target_sz, model, hp, device=device)  # init tracker
            elif end_frame >= f > start_frame:  # tracking
                state = siamese_track(state, im, mask_enable, refine_enable, device=device)  # track
                mask = state['mask']
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                pred_masks[obj_id, f, :, :] = mask
    toc /= cv2.getTickFrequency()

    if len(annos) == len(image_files):
        multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
                                            start=video['start_frame'] if 'start_frame' in video else None,
                                            end=video['end_frame'] if 'end_frame' in video else None)
        for i in range(object_num):
            for j, thr in enumerate(thrs):
                logger.info(
                    'Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
                                                                             multi_mean_iou[i, j]))
    else:
        multi_mean_iou = []

    if args.save_mask:
        video_path = join('test', args.dataset, 'SiamMask', video['name'])
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
          np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for i in range(pred_mask_final.shape[0]):
            cv2.imwrite(join(video_path, image_files[i].split('/')[-1].split('.')[0] + '.png'),
                        pred_mask_final[i].astype(np.uint8))

    if args.visualization:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
          np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[pred_mask_final]
        for f, image_file in enumerate(image_files):
            output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f, :, :, :])).astype("uint8")
            cv2.imshow("mask", output)
            cv2.waitKey(1)

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f * len(object_ids) / toc))

    return multi_mean_iou, f * len(object_ids) / toc


def main():
    global args, logger, v_id
    args = parser.parse_args()
    cfg = load_config(args)

    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    # setup model
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        parser.error('invalid architecture: {}'.format(args.arch))

    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)
    model.eval()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    model = model.to(device)
    # setup dataset
    dataset = load_dataset(args.dataset)

    # VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output
    else:
        vos_enable = False

    total_lost = 0  # VOT
    iou_lists = []  # VOS
    speed_list = []

    for v_id, video in enumerate(dataset.keys(), start=1):
        if args.video != '' and video != args.video:
            continue

        if vos_enable:
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                        args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], device=device)
            iou_lists.append(iou_list)
        else:
            lost, speed = track_vot(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                    args.mask, args.refine, device=device)
            total_lost += lost
        speed_list.append(speed)

    # report final result
    if vos_enable:
        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info('Segmentation Threshold {:.2f} mIoU: {:.3f}'.format(thr, iou))
    else:
        logger.info('Total Lost: {:d}'.format(total_lost))

    logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


def append_pred(pred, frame, objectID, x, y, w, h):
    my_dict = {'FrameID': frame, 'ObjectID': objectID, 'x_topleft': x, 'y_topleft': y, 'Width': w, 'Height': h,
              'isActive': 1, 'isOccluded': 0}
    pred = pred.append(pd.DataFrame(my_dict, index=[0]))
    return pred


def create_colormap_hsv(num_col):
    if num_col == 1:
        colors = [(255, 0, 0)]
        return colors
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



if __name__ == '__main__':
    main()