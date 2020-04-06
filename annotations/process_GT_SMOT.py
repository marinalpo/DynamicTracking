import os
import numpy as np
import pandas as pd


def process_GT_SMOT(path, seq):

    # Open connection
    f = open(path + 'gt_old.txt', "r")

    # Number of objects tracked
    num_obj = int(f.readline())

    for i in range(num_obj):
        myDict = {}
        objectID, initFrame, endFrame = tuple(np.fromstring(f.readline(), dtype=int, sep=' '))
        myDict['FrameID'] = list(np.arange(initFrame, endFrame + 1))
        myDict['ObjectID'] = list(objectID * np.ones(endFrame-initFrame+1).astype(int))
        myDict['x_topleft'] = list(np.fromstring(f.readline(), dtype=float, sep=' '))
        myDict['y_topleft'] = list(np.fromstring(f.readline(), dtype=float, sep=' '))
        myDict['Width'] = list(np.fromstring(f.readline(), dtype=float, sep=' '))
        myDict['Height'] = list(np.fromstring(f.readline(), dtype=float, sep=' '))
        if seq == 'crowd':
            active = []
            for j in range(endFrame - initFrame + 1):
                a = f.readline()
                active.append(a[0])
            myDict['isActive'] = active
        else:
            myDict['isActive'] = np.fromstring(f.readline(), dtype=int, sep=' ')
        if i == 0:
            df_SMOT = pd.DataFrame(myDict)
        else:
            df_SMOT = df_SMOT.append(pd.DataFrame(myDict))

    # Close connection
    f.close()

    # Sort dataframe by FrameID
    df_SMOT = df_SMOT.sort_values('FrameID')

    # Add zero-valued column for 'isOccluded'
    df_SMOT['isOccluded'] = 0

    df_SMOT.head(2)

    # Save dataframe as .txt
    df_SMOT.to_csv(path + 'gt.txt', header=None, index=None, sep=',')


root = '/data/SMOT/'
sequences = sorted(os.listdir(root))  # ['acrobats', 'balls', ... ]

for s in range(1, len(sequences)):
    print('Sequence: ', sequences[s])
    process_GT_SMOT(root + sequences[s] + '/gt/', sequences[s])


    # # Rename Frames
    # frames = sorted(os.listdir(root + sequences[s] + '/img/'))
    # for f, frame in enumerate(frames):
    #     old = root + sequences[s] + '/img/' + frame
    #     new = root + sequences[s] + '/img/' + str(f+1).zfill(6) + '.jpg'
    #     os.rename(old, new)
