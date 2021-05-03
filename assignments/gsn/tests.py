from assignments.gsn.assig1 import CustomFunctional, ShapesDataset, Augmentations
import pandas as pd
import torch
torch.set_printoptions(linewidth=700)
"""60_sigmoid"""
# preoutputs = 3 * (torch.rand(2, 60) - 0.5)
# outputs = CustomFunctional._10_piecewise_softmax(preoutputs)
# print(preoutputs.shape, outputs.shape)
# print(preoutputs[0][0:10].sum().item(), preoutputs[0][10:20].sum().item())
# print(outputs[0][0:10].sum().item(), outputs[0][10:20].sum().item())

"""loss_count60"""
H = 0.6
l = 0.2

label = [4, 6, 0, 0, 0, 0]
rs =        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
js =        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
outputs =   [0, 0, 0, l, H, l, 0, 0, 0, 0, 0, 0, 0, 0, 0, l, l, H, 0, 0, H, l, l, 0, 0, 0, 0, 0, 0, 0, H, l, l, 0, 0, 0, 0, 0, 0, 0, H, l, l, 0, 0, 0, 0, 0, 0, 0, H, l, l, 0, 0, 0, 0, 0, 0, 0]
elems = [o * (r - j) ** 2 for o, r, j in zip(outputs, rs, js)]
df = pd.DataFrame().assign(RS=rs, JS=js, OUTPUTS=outputs, ELEMS=elems)
# print(len(rs))
# print(rs[9], rs[10], rs[19], rs[20])


labels_ = torch.Tensor(label).reshape(1, 6)
outputs_ = torch.Tensor(outputs).reshape(1, 60)
CustomFunctional.loss_count60(outputs_, labels_)

"""augmentations"""
# trainset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice_=slice(0, 90))
#
# _images = Augmentations.augment_image(trainset.images[9])
# _labels = Augmentations.augment_label(trainset.labels[9])
# LIMIT = len(_images)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 4)
# for _img, _lab, _ax in zip(_images, _labels[:LIMIT], ax.flatten()[:LIMIT]):
#     _ax.imshow(_img[0, :, :].numpy())
#     _ax.set_xlabel(str(_lab))