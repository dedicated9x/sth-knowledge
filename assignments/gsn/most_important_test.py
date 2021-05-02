import torch
import torch.nn as nn
from assignments.gsn.assig1 import CustomFunctional, ShapesDataset, Linear, Net, MnistTrainer


torch.manual_seed(0)
CustomFunctional.init()
path_to_data = rf"C:\Datasets\gsn-2021-1"
testset = ShapesDataset(path_to_data, slice(9000, 10000))
trainset = ShapesDataset(path_to_data, slice(0, 9000))
# trainset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice(0, 9000), augmented=True)
conv_arbitrary = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2), nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
mnistslayer_head = nn.Sequential(Linear(784, 64), nn.ReLU())
mnistslayer_body = nn.Sequential(Linear(64, 64), nn.ReLU())
dlast6 = Linear(64, 6)
dlast60 = Linear(64, 60)
dlast135 = Linear(64, 135)
CF = CustomFunctional
_updated = lambda x, y: dict(x, **y)
_dense = {'dfirst': mnistslayer_head, 'dcore': mnistslayer_body}
_convdens = _updated(_dense, {'conv': conv_arbitrary})
_convdens6 = _updated(_convdens, {'dlast': dlast6, 'nonlin_outlayer': torch.sigmoid})
_convdens60 = _updated(_convdens, {'dlast': dlast60, 'nonlin_outlayer': CF._10_piecewise_softmax})
_convdens135 = _updated(_convdens, {'dlast': dlast135, 'nonlin_outlayer': lambda outputs: torch.softmax(outputs, dim=1)})
net_base6 = Net(**_convdens6)
net_base60 = Net(**_convdens60)
net_base135 = Net(**_convdens135)
# net_base135.load_state_dict(torch.load(rf"C:\temp\output\state2.pickle"))
trainer_classify6 = MnistTrainer(datasets=(trainset, testset), loss=CF.loss_classify6, acc=CF.acctransform_classify6)
log_clf6 = trainer_classify6.train(net=net_base6, no_epoch=2, verbose=1)
trainer_count60 = MnistTrainer(datasets=(trainset, testset), loss=CustomFunctional.loss_count60, acc=CustomFunctional.acctransform_count60)
log_count60 = trainer_count60.train(net=net_base60, no_epoch=2, verbose=1)
trainer_count135 = MnistTrainer(datasets=(trainset, testset), loss=CustomFunctional.loss_count135, acc=CustomFunctional.acctransform_count135)
log_count135 = trainer_count135.train(net=net_base135, no_epoch=2, verbose=1, reset=True)

""" 
[1,    20] loss: 4.047
[1,    40] loss: 3.792
[1,    60] loss: 3.878
Accuracy of the network on the 1000 test images: 6.8 %
[2,    20] loss: 4.013
[2,    40] loss: 3.821
[2,    60] loss: 3.813
Accuracy of the network on the 1000 test images: 16.3 %
[1,    20] loss: 59.664
[1,    40] loss: 50.674
[1,    60] loss: 51.983
Accuracy of the network on the 1000 test images: 0.0 %
[2,    20] loss: 54.151
[2,    40] loss: 51.374
[2,    60] loss: 50.749
Accuracy of the network on the 1000 test images: 0.0 %
[1,    20] loss: 6.186
[1,    40] loss: 5.858
[1,    60] loss: 5.812
Accuracy of the network on the 1000 test images: 0.9 %
[2,    20] loss: 6.005
[2,    40] loss: 5.692
[2,    60] loss: 5.693
Accuracy of the network on the 1000 test images: 1.3 %
"""

