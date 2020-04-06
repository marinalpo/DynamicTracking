import os
import pandas as pd


def process_GT_MOT(path):

    columns_MOT = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'x_3D', 'y_3D',
                   'z_3D']

    # Load annotations as dataframe
    df_mot = pd.read_csv(path, sep=',', header=None)

    # Assign column names
    df_mot.columns = columns_MOT

    # Delete non useful columns
    df_mot = df_mot.drop(df_mot.columns[[7, 8, 9]], axis=1)

    # Add zero-valued column for 'isOccluded'
    df_mot['isOccluded'] = 0

    # Save dataframe as .txt
    df_mot.to_csv('gt.txt', header=None, index=None, sep=',')


root = '/data/2DMOT2015/train/'
sequences = sorted(os.listdir(root))

for s in range(1, len(sequences)):
    print('Sequence:', sequences[s])
    path = root + sequences[s] + '/gt/'
    # os.rename(path + 'gt.txt', path + 'gt_old.txt')
    process_GT_MOT(path + 'gt_old.txt')
