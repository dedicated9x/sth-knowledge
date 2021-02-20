# TODO pokazuje przyklad z subframe'a
import pathlib as pl
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from scipy.spatial.transform import Rotation as scipy_rotation
import numpy as np
from mini_projects.biwi.to_rem import get_coord_good, get_coord_bad

TEST_PATH = pl.Path(rf'C:\Users\devoted\Downloads\kinect_head_pose_db\hpdb')


def get_coordinates(row: pd.DataFrame):
    with open(row['annotation_path'], 'r') as infile:
        text = infile.read()
    # coords = [float(elem) for elem in re.findall("\d+\.\d+", text)]
    coords = [float(elem) for elem in re.findall("\-?\d+\.?\d+", text)]
    try:
        row['RX1'], row['RX2'], row['RX3'], row['RY1'], row['RY2'], row['RY3'], row['RZ1'], row['RZ2'], row['RZ3'], row['X'], row['Y'], row['Z'] = coords
    except ValueError:
        print('elo')
        pass  # ~70 annotations are invalid #33
    return row


def get_readable_coordinates(row: pd.DataFrame):
    R = np.array([
        [row['RX1'], row['RX2'], row['RX3']],
        [row['RY1'], row['RY2'], row['RY3']],
        [row['RZ1'], row['RZ2'], row['RZ3']]
    ])
    R = np.transpose(R)
    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi

    row['yaw'] = yaw
    row['pitch'] = pitch
    row['roll'] = roll
    return row

def get_image_paths(row: pd.DataFrame):
    image_name = row['annotation_path'].stem[:-4] + 'rgb'
    row['image_path'] = row['annotation_path'].with_name(image_name).with_suffix('.png')
    return row


root = TEST_PATH
# annotations_paths = list(root.glob('**/*.txt'))
# annotations_paths = annotations_paths[1:]  # 'readme.txt' excluded
# id_list = [path.parent.stem + path.stem[5:-5] for path in annotations_paths]
# df = pd.DataFrame().assign(id=id_list, annotation_path=annotations_paths)
# df = df.apply(get_coordinates, axis=1) # ~5 minutes



path_to_pickle = pl.Path(__file__).with_name('df.pickle')
df = pd.read_pickle(path_to_pickle)
df = df.apply(get_readable_coordinates, axis=1)
df = df.apply(get_image_paths, axis=1)


chosen_numbers = [3, 31, 107, 129, 144, 220, 291, 321, 342, 366, 418, 435, 447, 500]
f, axarr = plt.subplots(4, 4)
for idx, n in enumerate(chosen_numbers):
    row = df[df['id'] == '01_' + str(n).zfill(5)]
    image_path = row['image_path'].values[0]
    axarr[divmod(idx, 4)].imshow(mpimg.imread(image_path))
    label = '|'.join([str(round(elem, 1)) for elem in row[['yaw', 'pitch', 'roll']].values[0]])
    axarr[divmod(idx, 4)].set_ylabel(label)

# feature = 'yaw'
feature = 'pitch'
feature = 'yaw'
f, axarr = plt.subplots(4, 4)

def plot_image(row: pd.DataFrame):
    print(row.name)
    image_path = row['image_path']
    axarr[divmod(row.name, 4)].imshow(mpimg.imread(image_path))
    axarr[divmod(row.name, 4)].set_ylabel(int(row[feature]))


chosen_numbers = [3, 31, 107, 129, 144, 220, 291, 321, 342, 366, 418, 435, 447, 500]
id_list = ['01_' + str(n).zfill(5) for n in chosen_numbers]
rows = df[df['id'].isin(id_list)]
rows = rows.sort_values(by=[feature])
rows = rows.reset_index()
rows.apply(plot_image, axis=1)


CHOSEN_NUMBERS = [3, 31, 107, 129, 144, 220, 291, 321, 342, 366, 418, 435, 447, 500]
def print_feature(df_, feature):
    global CHOSEN_NUMBERS
    id_list = ['01_' + str(n).zfill(5) for n in CHOSEN_NUMBERS]
    rows = df_[df_['id'].isin(id_list)].sort_values(by=[feature]).reset_index()

    f, axarr = plt.subplots(4, 4)
    for index, row in rows.iterrows():
        image_path = row['image_path']
        axarr[divmod(index, 4)].imshow(mpimg.imread(image_path))
        axarr[divmod(index, 4)].set_ylabel(int(row[feature]))

    return rows

z1 = print_feature(df, 'yaw')
z1 = print_feature(df, 'pitch')
z1 = print_feature(df, 'roll')

for index, row in z1.iterrows():
    print(row['image_path'])

# def func_123(row):
#     yaw = row['yaw']
#     print(yaw)
#     # print('elo')
#     return 1
#
# rows.apply(func_123, axis=1)
#
#
#
#




"""test, czy dobrze parsuje"""
good = get_coord_good(df, '01_00144')
bad = get_coord_bad(df, '01_00144')
print(good)
print(bad)
