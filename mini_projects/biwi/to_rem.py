import numpy as np

def get_coord_good(df_, id_):
    # row = df[df['id'] == '01_00144']
    row = df_[df_['id'] == id_]
    pose_path = row['annotation_path'].values[0]

    pose_annot = open(pose_path, 'r')
    R = []
    for line in pose_annot:
        line = line.strip('\n').split(' ')
        l = []
        if line[0] != '':
            for nb in line:
                if nb == '':
                    continue
                l.append(float(nb))
            R.append(l)
    R = np.array(R)
    T = R[3,:]
    R = R[:3,:]
    pose_annot.close()

    R = np.transpose(R)
    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    return [yaw, pitch, roll]

def get_coord_bad(df_, id_):
    return list(df_[df_['id'] == id_][['yaw', 'pitch', 'roll']].values[0])
