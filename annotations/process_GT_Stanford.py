import os
import pandas as pd


def process_GT_Stanford(path):

    columns_stanford = ['ObjectID', 'xmin', 'ymin', 'xmax', 'ymax', 'FrameID', 'isLost', 'isOccluded', 'Generated',
                        'Label']
    columns_standard = ['FrameID', 'ObjectID', 'x_topleft', 'y_topleft', 'Width', 'Height', 'isActive', 'isOccluded']

    # Load annotations as dataframe
    df_stan = pd.read_csv(path, sep=' ', header=None)

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
    df_stan = df_stan[df_stan.FrameID < 400]

    # Save dataframe as .txt
    df_stan.to_csv(path + 'gt.txt', header=None, index=None, sep=',')


root = '/data/2DMOT2015/train/'
sequences = sorted(os.listdir(root))  # ['acrobats', 'balls', ... ]

for s in range(1, len(sequences)):
    print('Sequence:', sequences[s])
    path = root + sequences[s] + '/gt/'
    # os.rename(path + 'gt.txt', path + 'gt_old.txt')
    process_GT_Stanford(path + 'gt_old.txt')