# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path /data/Ponc/tracking/JPEGImages/480p/nhl/
# --------------------------------------------------------
import glob
import pickle 
from tools.test_raw import *
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from shapely.geometry import Polygon
from shapely.geometry import asPolygon
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/balls', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    
    # Parse Image file
    # img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))[772:805] # NHL
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))[32:88] # NFL
    
    ims = [cv2.imread(imf) for imf in img_files]
    
    try:
        # init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        init_rect = (737,165, 76, 76) # balls noia
        init_rect = (75,388, 200, 79) # Seagull
        # init_rect = (536,105, 41, 40) # juggling-easy
        init_rect = (437, 306, 115,50) # Eagles
        init_rect = (737, 374, 100, 156) # NHL
        init_rect = (1027, 259, 57, 100) # NFL
        x, y, w, h = init_rect

    except:
        exit()

    toc = 0
    print("num images ",len(ims))


    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        frame_boxes = []
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            # state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            laloc = np.int0(location).reshape((-1,1,2))
            traj = np.int0(np.average(laloc, axis=0)[0])
            frame_boxes.append([traj])
            for i in range(laloc.shape[0]):
                cv2.circle(im, (laloc[i,0,0], laloc[i,0,1]),6,(0,0,255))
                cv2.circle(im, (traj[0], traj[1]) , 3, (255,255,0), 2)
            
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.putText(im,str(state['score']),(50,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0))
            cv2.imwrite('/data/Ponc/tracking/results/crossing_football_superbowl_2020/'+str(f)+'.jpeg', im)

        toc += cv2.getTickCount() - tic

    
    toc /= cv2.getTickFrequency()
    fps = f / toc
    
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))