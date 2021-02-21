import re
import pathlib as pl
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_rotation
from mini_projects.biwi.to_rem import get_coord_good, get_coord_bad
from mini_projects.biwi.feature_visualizer import Visualizer

TEST_PATH = pl.Path(rf'C:\Users\devoted\Downloads\kinect_head_pose_db\hpdb')

def get_paths(root_path):
    annotations_paths = list(root_path.glob('**/*.txt'))[1:]  # 'readme.txt' excluded
    image_paths = [path.with_name(path.stem[:-4] + 'rgb').with_suffix('.png') for path in annotations_paths]
    ids = [path.parent.stem + path.stem[5:-5] for path in annotations_paths]
    return {'id': ids, 'annotation_path': annotations_paths, 'image_path': image_paths}
#
# def get_image_paths(row: pd.DataFrame):
#     image_name = row['annotation_path'].stem[:-4] + 'rgb'
#     row['image_path'] = row['annotation_path'].with_name(image_name).with_suffix('.png')
#     return row

def get_coordinates_v2(row: pd.DataFrame):
    with open(row['annotation_path'], 'r') as infile:
        text = infile.read()
    coords = [float(elem) for elem in re.findall("\-?\d+\.?\d+", text)]
    try:
        row['rot'], row['X'], row['Y'], row['Z'] = coords[:9], coords[9], coords[10], coords[11]
    except IndexError:
        pass  # ~2 annotations are invalid
    return row


def get_readable_coordinates_v2(row: pd.DataFrame):
    R = np.array(row['rot']).reshape(3, 3)
    R = np.transpose(R)
    #
    # yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    # pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
    # roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    # #
    # row['yaw'] = yaw
    # row['pitch'] = pitch
    # row['roll'] = roll

    # pitch, yaw, roll = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
    # pitch, yaw, roll = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
    # pitch2, yaw2, roll2 = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
    row['pitch'], row['yaw'], row['roll'] = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
    # row['yaw'] = yaw2
    # row['pitch'] = pitch2
    # row['roll'] = roll2

    return row


root = TEST_PATH
df = pd.DataFrame().assign(**get_paths(root))
df = df.apply(get_coordinates_v2, axis=1) # ~5 minutes
df = df[df['rot'].apply(lambda x: type(x)) == list]
df = df.apply(get_readable_coordinates_v2, axis=1)


# TODO refaktor na malej ilosc
df = df.drop(['yaw', 'pitch', 'roll'], axis=1)


""" zamiana na scipy """
# row = df[df['id'] == '01_00144']
# R = np.array(row['rot'].values[0]).reshape(3, 3)
# R = np.transpose(R)
#
# yaw1 = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
# pitch1 = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
# roll1 = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
# [yaw1, pitch1, roll1]
#
# # r = scipy_rotation.from_matrix(R)
# pitch, yaw, roll = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
# [yaw, pitch, roll]

""" pickling """
# path_to_pickle = pl.Path(__file__).with_name('df.pickle')
# df.to_pickle(path_to_pickle.with_name('df2.pickle'))
# df1 = pd.read_pickle(path_to_pickle.with_name('df2.pickle'))

"""wizualizacja feature'a"""
z1 = Visualizer.plot_feature(df, 'yaw')

"""test, czy dobrze parsuje"""
good = get_coord_good(df, '01_00144')
bad = get_coord_bad(df, '01_00144')
print(good)
print(bad)
