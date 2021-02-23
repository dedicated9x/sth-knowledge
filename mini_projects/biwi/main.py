import re
import pathlib as pl
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_rotation
from mini_projects.biwi.feature_visualizer import Visualizer

TEST_PATH = pl.Path(rf'C:\Users\devoted\Downloads\kinect_head_pose_db\hpdb')

def get_paths(root_path):
    annotations_paths = list(root_path.glob('**/*.txt'))[1:]  # 'readme.txt' excluded
    image_paths = [path.with_name(path.stem[:-4] + 'rgb').with_suffix('.png') for path in annotations_paths]
    ids = [path.parent.stem + path.stem[5:-5] for path in annotations_paths]
    return {'id': ids, 'annotation_path': annotations_paths, 'image_path': image_paths}

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
    row['pitch'], row['yaw'], row['roll'] = scipy_rotation.from_matrix(R).as_euler('xyz', degrees=True) * [1, -1, -1]
    return row


root = TEST_PATH
df = pd.DataFrame().assign(**get_paths(root))
df = df.apply(get_coordinates_v2, axis=1) # ~5 minutes
df = df[df['rot'].apply(lambda x: type(x)) == list]
df = df.apply(get_readable_coordinates_v2, axis=1)



"""wizualizacja feature'a"""
z1 = Visualizer.plot_feature(df, 'yaw')
