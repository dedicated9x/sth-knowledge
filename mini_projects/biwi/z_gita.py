import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from PIL import Image, ImageFilter

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.png', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext))
        img = img.convert(self.image_mode)
        pose_path = os.path.join(self.data_dir, self.y_train[index] + '_pose' + self.annot_ext)

        y_train_list = self.y_train[index].split('/')
        bbox_path = os.path.join(self.data_dir, y_train_list[0] + '/dockerface-' + y_train_list[-1] + '_rgb' + self.annot_ext)

        # Load bounding box
        bbox = open(bbox_path, 'r')
        line = bbox.readline().split(' ')
        if len(line) < 4:
            x_min, y_min, x_max, y_max = 0, 0, img.size[0], img.size[1]
        else:
            x_min, y_min, x_max, y_max = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        bbox.close()

        # Load pose in degrees
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

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        # Loosely crop face
        k = 0.35
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length