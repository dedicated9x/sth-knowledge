import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import functools
import pandas as pd
import pathlib as pl
from torch.utils.data import Dataset
from torchvision.io import read_image
import sys
import itertools


class Linear(torch.nn.Module):
    """in_features, out_features -> <liczba> neuronów przed, <liczba> neuronów po tej warstwie."""
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_in')
        init.zeros_(self.bias)

    def forward(self, x):
        r = x.matmul(self.weight.t()) # .t() => wyciąga z obiektu Parameter, jego obiekt bazowy Tensor (obliczenia tego wymagają)
        r += self.bias
        return r

class Net(nn.Module):
    def __init__(self, dfirst, dcore, dlast, nonlin_outlayer=None, conv=None):
        super(Net, self).__init__()
        self.conv = conv if conv is not None else torch.nn.Identity()
        self.dense_first = dfirst
        self.dense_core = dcore
        self.dense_last = dlast
        self.nonlin_outlayer = nonlin_outlayer if nonlin_outlayer is not None else torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 28 * 28)
        x = self.dense_first(x)
        x = self.dense_core(x)
        x = self.dense_last(x)
        x = self.nonlin_outlayer(x)
        x = self.interiorize(x)
        return x

    def interiorize(self, tensor_):
        eps = 1e-4
        return (1 - 2 * eps) * tensor_ + eps

# TODO on tez do wyjebki
MB_SIZE = 128

class ShapesDataset(Dataset):
    def __init__(self, root, slice_=None, augmented=False, transform=None, target_transform=None):
        self.img_dir = pl.Path(root).joinpath('data')
        self.df = pd.read_csv(self.img_dir.joinpath('labels.csv'))
        if slice_ is not None:
            self.df = self.df[slice_]
        self.images = torch.stack([read_image(self.img_dir.joinpath(name).__str__())[0:1] / 255 for name in self.df['name']])
        self.labels = torch.tensor(self.df.drop(['name'], axis=1).values)

        if augmented:
            self.labels = torch.cat([torch.stack(Augmentations.augment_label(l)) for l in self.labels], dim=0)
            self.images = torch.cat([torch.stack(Augmentations.augment_image(im)) for im in self.images], dim=0)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample

class Augmentations:
    indices = torch.tensor([
        [0, 1, 2, 3, 4, 5],  # base
        [0, 1, 5, 2, 3, 4],  # 90 right
        [0, 1, 4, 5, 2, 3],  # 180 right
        [0, 1, 3, 4, 5, 2],  # 270 right
        [0, 1, 2, 5, 4, 3],  # vertical flip
        [0, 1, 3, 2, 5, 4],  # vertical flip + 90 right
        [0, 1, 4, 3, 2, 5],  # vertical flip + 180 right
        [0, 1, 5, 4, 3, 2],  # vertical flip + 270 right
    ]).flatten()

    @classmethod
    def augment_label(cls, label):
        return list(label.index_select(0, cls.indices).split(6))

    @staticmethod
    def augment_image(image):
        image_vert_flip = image.flip(2)
        images = [
            image,                              # base
            image.rot90(3, [1, 2]),             # 90 right
            image.rot90(2, [1, 2]),             # 180 right
            image.rot90(1, [1, 2]),             # 270 right
            image_vert_flip,                    # vertical flip
            image_vert_flip.rot90(3, [1, 2]),   # vertical flip + 90 right
            image_vert_flip.rot90(2, [1, 2]),   # vertical flip + 180 right
            image_vert_flip.rot90(1, [1, 2])    # vertical flip + 270 right
        ]
        return images


class MnistTrainer(object):
    def __init__(self, datasets, loss, acc, ):
        # self.net = net
        # self.no_epoch = no_epoch
        self.trainset, self.testset = datasets
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=MB_SIZE, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False)
        self.loss = loss
        self.accuracy = acc
        self.print_period = 20

    def train(self, net, no_epoch=20):
        """FOCUS: sgd dostaje info o sieci, jaką będzie trenował"""
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
        log_acc = []

        for epoch in range(no_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                """pytorch z defaultu sumuje (!) dotychczasowe gradienty. Tym je resetujemy."""
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                """tensor (loss) liczy TYLKO 'grad'.  A przecież nam zależy na minimum (w końcu SDG)"""
                optimizer.step()

                """+= -> bo chcemy logowac troche wieksze liczby"""
                running_loss += loss.item()
                if ((i != 0) * (i + 1)) % self.print_period == 1:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / self.print_period))
                    running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.testloader:
                    inputs_, labels_ = data
                    outputs_ = net(inputs_)
                    # correct += self.accuracy(outputs_, labels_)
                    outputs_transformed, labels_transformed = self.accuracy(outputs_, labels_)
                    correct += self.count_matches(outputs_transformed, labels_transformed)
                    total += outputs_.shape[0]
            accuracy = 100 * correct / total
            log_acc.append(accuracy)
            print('Accuracy of the network on the {} test images: {} %'.format(total, accuracy))
        return log_acc

    def count_matches(self, outputs_transformed, labels_transformed):
        """outputs & labels MUST be integer tensors"""
        no_correct = (outputs_transformed == labels_transformed).all(dim=1).int().sum().item()
        return no_correct

class CustomFunctional:
    counts2class = None # Will be calculated in external scope

    @classmethod
    def init(cls):
        pairs = [[1, 9], [2, 8], [3, 7], [4, 6], [5, 5]]
        counts = list(set(itertools.chain(*[itertools.permutations(p + [0, 0, 0, 0]) for p in pairs])))
        cls.counts2class = {count: i for i, count in enumerate(counts)}

    @staticmethod
    def binarize_topk(batch, k):
        return F.one_hot(torch.topk(batch, k).indices, batch.shape[1]).sum(dim=1)

    @staticmethod
    def loss_nll(outputs, labels):
        neg_sums = -torch.sum(torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels), dim=1)
        loss = torch.mean(neg_sums)
        return loss

    @ staticmethod
    def loss_count60(outputs, labels):
        rs = labels.repeat_interleave(10, dim=1)
        js = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).repeat(6)
        loss = (outputs * (rs - js) ** 2).sum(dim=1).mean()
        return loss

    @ staticmethod
    def acctransform_count60(outputs, labels):
        outputs_ = torch.stack(outputs.split(10, dim=1)).argmax(dim=2).T
        return outputs_, labels

    @ staticmethod
    def _10_piecewise_softmax(outputs):
        return torch.stack(outputs.split(10, dim=1)).softmax(dim=2).transpose(0, 1).flatten(1, 2)

    @ classmethod
    def loss_classify6(cls, outputs, labels):
        # TODO binarize topk_tutaj wloz
        labels_ = cls.binarize_topk(labels, 2)
        loss = cls.loss_nll(outputs, labels_)
        return loss

    @ classmethod
    def acctransform_classify6(cls, outputs, labels):
        outputs_bin = cls.binarize_topk(outputs, 2)
        labels_bin = cls.binarize_topk(labels, 2)
        return outputs_bin, labels_bin

    @ staticmethod
    def loss_count135(outputs, labels):
        labels_ = torch.tensor([CustomFunctional.counts2class[tuple(e.numpy())] for e in labels])
        loss = CustomFunctional.loss_nll(outputs, F.one_hot(labels_, 135))
        return loss

    @ classmethod
    def acctransform_count135(cls, outputs, labels):
        outputs_transformed = torch.topk(outputs, 1).indices
        labels_ = torch.tensor([CustomFunctional.counts2class[tuple(e.numpy())] for e in labels])
        labels_transformed = labels_.unsqueeze(dim=1)
        return outputs_transformed, labels_transformed


class Utils:
    @staticmethod
    def get_loss_inputs(trainset, net, mb_size):
        loader = torch.utils.data.DataLoader(trainset, batch_size=mb_size, shuffle=False)
        inputs, labels = next(loader.__iter__())
        outputs = net(inputs)
        return outputs, labels

    @staticmethod
    def get_acc_inputs(testset, net, mb_size):
        return Utils.get_loss_inputs(testset, net, mb_size)

    @staticmethod
    def plot_8images(_images):
        _labels = ['None'] * 8
        LIMIT = len(_images)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4)
        for _img, _lab, _ax in zip(_images, _labels[:LIMIT], ax.flatten()[:LIMIT]):
            _ax.imshow(_img[0, :, :].numpy())
            _ax.set_xlabel(str(_lab))

    @staticmethod
    def conv_output_shape(net_):
        output = net_(torch.zeros([1, 1, 28, 28]))
        return output.shape, output.flatten().shape

REF = {}

def func1(x):
    a = 1
    return x

def main():
    torch.manual_seed(0)
    CustomFunctional.init()
    # return

    trainset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice(0, 9000))
    # trainset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice(0, 9000), augmented=True)
    testset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice(9000, 10000))
    # testset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice(9000, 10000), augmented=True)

    REF['TRAINSET'] = trainset
    REF['TESTSET'] = testset
    # return

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

    REF['NET'] = conv_arbitrary
    # return


    # net_base135.load_state_dict(torch.load(rf"C:\temp\output\state2.pickle"))

    # TODO przepisac dwa pozostale
    # TODO verbose
    # TODO zrobic taka architekture, aby byl init i reinit ----------> reset_parameters jako atrybut w "train" HEHE!!!
    trainer_classify6 = MnistTrainer(datasets=(trainset, testset), loss=CF.loss_classify6, acc=CF.acctransform_classify6)
    log_clf6 = trainer_classify6.train(net=net_base6, no_epoch=2)
    # trainer = MnistTrainer(net=net_base60, datasets=(trainset, testset), loss=CustomFunctional.loss_count60, acc=CustomFunctional.acctransform_count60, no_epoch=2)
    # trainer.train()
    # trainer = MnistTrainer(net=net_base135, datasets=(trainset, testset), loss=CustomFunctional.loss_count135, acc=CustomFunctional.acctransform_count135, no_epoch=2)
    # trainer.train()
    trainer_count60 = MnistTrainer(datasets=(trainset, testset), loss=CustomFunctional.loss_count60, acc=CustomFunctional.acctransform_count60)
    log_count60 = trainer_count60.train(net=net_base60, no_epoch=2)
    trainer_count135 = MnistTrainer(datasets=(trainset, testset), loss=CustomFunctional.loss_count135, acc=CustomFunctional.acctransform_count135)
    log_count135 = trainer_count135.train(net=net_base135, no_epoch=20)

    

    # torch.save(net_base135.state_dict(), rf"C:\temp\output\state2.pickle")


""" 
[1,    20] loss: 4.047
[1,    40] loss: 3.792
[1,    60] loss: 3.878
Accuracy of the network on the 1000 test images: 6.8 %
[2,    20] loss: 4.013
[2,    40] loss: 3.821
[2,    60] loss: 3.813
Accuracy of the network on the 1000 test images: 16.3 %
"""
if __name__ == '__main__':
    main()




"""augmentations"""
# testset = REF['TESTSET']
#
# # _images = testset.images[8:16]
# _images = BUFF[555]
# _labels = ['None'] * 8
# LIMIT = len(_images)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 4)
# for _img, _lab, _ax in zip(_images, _labels[:LIMIT], ax.flatten()[:LIMIT]):
#     _ax.imshow(_img[0, :, :].numpy())
#     _ax.set_xlabel(str(_lab))



"""SCRATCH"""
# trainset = REF['TRAINSET']
# labels_aug = torch.cat([torch.stack(Augmentations.augment_label(l)) for l in trainset.labels], dim=0)
# images_aug = torch.cat([torch.stack(Augmentations.augment_image(im)) for im in trainset.images], dim=0)
#




"""test augmentacji na calym datasecie"""
# choice = torch.tensor([12, 43, 854, 23, 504, 203, 205, 289])
# _images = images_aug.index_select(0, choice)
# _labels = labels_aug.index_select(0, choice)
# LIMIT = len(_images)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 4)
# for _img, _lab, _ax in zip(_images, _labels[:LIMIT], ax.flatten()[:LIMIT]):
#     _ax.imshow(_img[0, :, :].numpy())
#     _ax.set_xlabel(str(_lab))







"""tworzenie funkcji"""
# torch.set_printoptions(linewidth=700)
# outputs, labels = Utils.get_acc_inputs(REF['TRAINSET'], REF['NET'], 4)


"""df2labels"""
# root = rf"C:\Datasets\gsn-2021-1"
# img_dir = pl.Path(root).joinpath('data')
# df = pd.read_csv(img_dir.joinpath('labels.csv'))
# lablen = 10





"""DECYZJA - tryb szybki"""
# Nie robimy, bo datasety za chwilę i tak będa niewielkie.



"""Jak ma być"""
#             (basic label)                       MA BYC OSTATECZNIE
# MNIST       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]      <- tak
# GSN         [0, 0, 0, 6, 0, 0, 4, 0, 0, 0]      [0, 0, 0, 6, 0, 0, 4, 0, 0, 0] lub [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]


"""skurwialy case 389"""
# labels = torch.tensor([5], dtype=torch.long) # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# outputs = torch.Tensor([[-18.5508, -17.9211, -19.8926, -13.2589, -16.2123,  16.8303, -14.1339, -20.7289, -11.2858, -14.3649]]) #torch.Size([2, 10])
#
#
# labels1 = F.one_hot(labels, 10) # torch.Size([2, 10])
# outputs1 = 0.9999 * torch.sigmoid(outputs)
# loss_nll(outputs1, labels1)
#
# outputs = outputs1
# labels = labels1


"""syntetyk"""
# outputs = torch.Tensor([[0.05, 0.9, 0.3, 0.05, 0.05, 0.05], [0.01, 0.95, 0.05, 0.3, 0.05, 0.05]]) #torch.Size([2, 6])
# labels = torch.Tensor([[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]]) #torch.Size([2, 6])


"""natural (stary init)"""
# labels = torch.tensor([6, 2], dtype=torch.long) # torch.Size([2])
# outputs = torch.Tensor([[47.0333, -8.8092, -31.7119, -15.8912, -14.0070, 58.2908, 15.3053, 16.3086, 14.9250, 1.6518], [ 19.9257,  -6.3817, -14.2751,  21.4164, -21.7247,  49.5124,  -2.5922, -21.6784,  -1.3972,  10.1371]]) #torch.Size([2, 10])


"""natural (nowy init)"""
# labels = torch.tensor([2, 5], dtype=torch.long) # torch.Size([2])
# outputs = torch.Tensor([[-0.0017,  1.1775, -0.3094, -0.3539,  0.0291, -0.0736, -0.1235,  0.0097, -0.9908,  0.5361], [ 0.1747,  0.5950,  0.0957, -0.3287, -0.1601, -0.0745,  0.2418,  0.0014, -0.3589,  0.3077]]) #torch.Size([2, 10])


"""Z czym był problem"""
# torch.log(1 - torch.sigmoid(torch.tensor([47.])))

"""testy log sumy"""
# outputs = torch.Tensor([[0.05, 0.9, 0.3, 0.05, 0.05, 0.05], [0.01, 0.95, 0.05, 0.3, 0.05, 0.05]]) #torch.Size([2, 6])
# labels = torch.Tensor([[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]]) #torch.Size([2, 6])
# y = labels
# y1 = outputs
# -torch.sum(torch.log(y1) * y + torch.log(1 - y1) * (1 - y))


"""XYZ Dzialanie (obecnych) loss funkcji"""
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# output = torch.Tensor([[47.0333, -8.8092, -31.7119, -15.8912, -14.0070, 58.2908, 15.3053, 16.3086, 14.9250, 1.6518]]) #torch.Size([1, 10])
# labels = torch.tensor([6], dtype=torch.long) # torch.Size([1])
# criterion(output, labels)


"""SHAPE - DEPENDPEND func"""
# def func1(output, target):
#     loss = torch.sum(torch.mean(output, dim=1) * target)
#     return loss
#
# output = torch.Tensor([[2., 3.]])
# target = torch.Tensor([1.2])
#
# outputS = torch.Tensor([[2., 3.], [4., 7.]])
# targetS = torch.Tensor([1.2, 0.5])
#
# func1(output, target)
# func1(outputS, targetS)


"""DECYZJA - custom"""
# trzeba zrobic recznie, bo i tak dostaniemy jakas sztuczna duplikacje #torch.Size([1, 6, 2]) + nie wiadomo, czy takie sumowanie ma sens


"""N i C"""
#torch.Size([batch_size,    y_size  ])
#torch.Size([N,             C       ])


"""INPUT, OUTPUT, TARGET, LABELS"""
# criterion(output, labels)
# loss(input, target)



