from assignments.gsn.assig1 import CustomFunctional, ShapesDataset, Augmentations
import pandas as pd
import torch

"""60_sigmoid"""
# preoutputs = 3 * (torch.rand(2, 60) - 0.5)
# print(preoutputs.shape)
#
# preoutput0 = preoutputs[0]
# print(preoutput0.shape)
# print(pd.Series(preoutput0.numpy()).describe())
#
# outputs = CustomFunctional._10_piecewise_softmax(preoutputs)
#
# print(outputs.shape)
# print(outputs[0][0:10].sum().item(), outputs[0][10:20].sum().item())


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