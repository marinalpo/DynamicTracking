import os
import pandas as pd
# from utils_annotation import *

dataset = 2  # 0: MOT, 1: SMOT, 2: stanford-campus

if dataset == 0:
    root = '/data/2DMOT2015/train/'
    sequences = sorted(os.listdir(root))
    for s in range(1, len(sequences)):
        print('Sequence:', sequences[s])
        path = root + sequences[s] + '/gt/'
        # os.rename(path + 'gt.txt', path + 'gt_old.txt')
        process_GT_MOT(path)
        create_init(path)

elif dataset == 1:
    root = '/data/SMOT/'
    sequences = sorted(os.listdir(root))  # ['acrobats', 'balls', ... ]
    for s in range(1, len(sequences)):
        print('Sequence: ', sequences[s])
        process_GT_SMOT(root + sequences[s] + '/gt/', sequences[s])
        create_init(root + sequences[s] + '/gt/')
        # Rename Frames
        frames = sorted(os.listdir(root + sequences[s] + '/img/'))
        for f, frame in enumerate(frames):
            old = root + sequences[s] + '/img/' + frame
            new = root + sequences[s] + '/img/' + str(f+1).zfill(6) + '.jpg'
            os.rename(old, new)

elif dataset == 2:
    root = '/data/stanford-campus/'
    anot = root + 'annotations/'
    sequences = sorted(os.listdir(anot))  # ['bookstore', ... ]
    for s in range(1, len(sequences)):
        print('Sequence:', sequences[s])
        path = anot + sequences[s] + '/'
        videos = sorted(os.listdir(path))
        for v in range(1, len(videos)):
            print('Video:', videos[v])
            old = path + videos[v] + '/annotations.txt'
            new = path + videos[v] + '/gt_old.txt'
            os.rename(old, new)
            process_GT_Stanford(path + videos[v] + '/')
            create_init(path + videos[v] + '/')