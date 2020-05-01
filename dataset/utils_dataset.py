import pandas as pd
import numpy as np
import pickle as pkl

def create_init(path):
    columns_standard = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive',
                        'isOccluded', 'cx', 'cy']

    gt = pd.read_csv(path + 'gt.txt', sep=',', header=None)
    gt.columns = columns_standard

    list_objectIDs = gt.ObjectID.unique()

    for count, o in enumerate(list_objectIDs):

        obj = gt[(gt.ObjectID == o)].reset_index(drop=True)
        # If isAcvive == -1 -> Object has reappeared
        dif = obj.diff(periods=-1)
        reap = obj.iloc[dif[dif.isActive == -1].index + 1]

        obj_first = gt[(gt.ObjectID == o) & (gt.isActive == 1) & (gt.isOccluded == 0)]

        if count == 0:
            init = obj_first.iloc[0:1]
        else:
            init = init.append(obj_first.iloc[0:1])
        init = init.append(reap)

    # Sort dataframe by FrameID
    init = init.sort_values('FrameID')

    # # Save dataframe as .txt
    init.to_csv(path + 'init.txt', header=None, index=None, sep=',')


def process_GT_MOT(path):

    columns_MOT = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'x_3D', 'y_3D',
                   'z_3D']

    # Load annotations as dataframe
    df_mot = pd.read_csv(path + 'gt_old.txt', sep=',', header=None)

    # Assign column names
    df_mot.columns = columns_MOT

    # Delete non useful columns
    df_mot = df_mot.drop(df_mot.columns[[7, 8, 9]], axis=1)

    # Add zero-valued column for 'isOccluded'
    df_mot['isOccluded'] = 0

    # Add centroids as columns
    df_mot['cx'] = (df_mot['x_topleft'] + df_mot['Width'] / 2)
    df_mot['cy'] = (df_mot['y_topleft'] + df_mot['Height'] / 2)

    # Save dataframe as .txt
    df_mot.to_csv(path + 'gt.txt', header=None, index=None, sep=',')


def process_GT_Stanford(path):

    columns_stanford = ['ObjectID', 'xmin', 'ymin', 'xmax', 'ymax', 'FrameID', 'isLost', 'isOccluded', 'Generated',
                        'Label']
    columns_standard = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'isOccluded']

    # Load annotations as dataframe
    df_stan = pd.read_csv(path + 'gt_old.txt', sep=' ', header=None)

    # Assign column names
    df_stan.columns = columns_stanford

    # Create Height and Width columns
    df_stan['Width'] = df_stan['xmax'] - df_stan['xmin']
    df_stan['Height'] = df_stan['ymax'] - df_stan['ymin']

    # Rename columns
    df_stan.rename(columns={'xmin': 'x_topleft'}, inplace=True)
    df_stan.rename(columns={'ymin': 'y_topleft'}, inplace=True)

    # Invert isLost column
    df_stan['isActive'] = df_stan['isLost'].replace({0: 1, 1: 0})

    # Delete non useful columns
    df_stan = df_stan.drop(df_stan.columns[[3, 4, 6, 8, 9]], axis=1)

    # Sort dataframe by FrameID and delay frames
    df_stan = df_stan.sort_values('FrameID')
    df_stan['FrameID'] = df_stan['FrameID'] + 1

    # Reorder columns
    df_stan = df_stan[columns_standard]

    # Delete annotations from frames after 400 (there are annotations for 13.334 frames!)
    # df_stan = df_stan[df_stan.FrameID < 400]

    # Add centroids as columns
    df_stan['cx'] = (df_stan['x_topleft'] + df_stan['Width'] / 2)
    df_stan['cy'] = (df_stan['y_topleft'] + df_stan['Height'] / 2)

    # Save dataframe as .txt
    df_stan.to_csv(path + 'gt.txt', header=None, index=None, sep=',')


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

    # Add centroids as columns
    df_SMOT['cx'] = (df_SMOT['x_topleft'] + df_SMOT['Width'] / 2)
    df_SMOT['cy'] = (df_SMOT['y_topleft'] + df_SMOT['Height'] / 2)

    # Save dataframe as .txt
    df_SMOT.to_csv(path + 'gt.txt', header=None, index=None, sep=',')
