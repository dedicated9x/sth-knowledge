import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import pathlib as pl
from torch.utils.data import Dataset
from torchvision.io import read_image

import sys


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class Linear(torch.nn.Module):
    """in_features, out_features -> <liczba> neuronów przed, <liczba> neuronów po tej warstwie."""
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        # truncated_normal_(self.weight, std=0.5)
        init.kaiming_normal_(self.weight, mode='fan_in')
        init.zeros_(self.bias)

    """x -> inputy (wchodzą w petli SGD)"""
    def forward(self, x):
        """.t() => wyciąga z obiektu Parameter, jego obiekt bazowy Tensor (obliczenia tego wymagają)"""
        r = x.matmul(self.weight.t())
        r += self.bias
        return r


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(784, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 10)

    """x -> inputy (wchodzą w petli SGD)"""
    def forward(self, x):
        """.view() to .reshape() dla tensorów"""
        x = x.view(-1, 28 * 28)
        """nn.Model.__call__() odpala .forward()"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


MB_SIZE = 128


class ShapesDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, slice_=None):
        self.img_dir = pl.Path(root).joinpath('data')
        img_labels = pd.read_csv(self.img_dir.joinpath('labels.csv'))
        if slice_ is not None:
            img_labels = img_labels[slice_]
        self.images = [read_image(self.img_dir.joinpath(name).__str__()) / 255 for name in img_labels['name']]
        self.labels = img_labels['label'].reset_index(drop=True)
        self.transform = None
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

def multiindex_nll_loss(outputs, labels):
    # loss = -torch.sum(torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels))
    loss = torch.mean(-torch.sum(torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels), dim=1))
    return loss

class MnistTrainer(object):
    def __init__(self, net, datasets, no_epoch=20):
        self.net = net
        self.no_epoch = no_epoch
        self.trainset, self.testset = datasets
        nw = lambda x: 0 if type(x) == ShapesDataset else 4
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=MB_SIZE, shuffle=True, num_workers=nw(self.trainset))
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=nw(self.testset))

    def train(self):
        """net -> to prostu nasza sieć (nn.Model)"""
        # net = Net()
        net = self.net
        """criterion -> zwykła funkcja, której użyjemy później"""
        criterion = nn.CrossEntropyLoss()
        """FOCUS: sgd dostaje info o sieci, jaką będzie trenował"""
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

        for epoch in range(self.no_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                """pytorch z defaultu sumuje (!) dotychczasowe gradienty. Tym je resetujemy."""
                optimizer.zero_grad()

                """nn.Model.__call__() odpala .forward()"""
                outputs = net(inputs)
                """label(1), output(10)"""
                """labels.shape -> torch.Size([128]) // outputs.shape -> torch.Size([128, 10])"""
                """criterion(outputs[0:1], labels[0:1])"""
                # loss = criterion(outputs, labels)

                labels1 = F.one_hot(labels, 10)  # torch.Size([2, 10])
                outputs1 = 0.9999 * torch.sigmoid(outputs)
                loss = multiindex_nll_loss(outputs1, labels1)
                #
                # labels = labels1
                # outputs = outputs1
                #
                # z1 = -torch.sum(torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels), dim=1)
                # z1[65]
                #
                # outputs = outputs[65:66]
                # labels = labels[65:66]
                #
                # z2 = -torch.sum(torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels), dim=1)
                # z2
                #
                #
                # if i == 389:
                #     a = 2

                # 0.24291181564331055
                # 0.21021124720573425
                # 0.2863709628582001
                # if 0.28637096 < loss.item() <0.28637097:
                #     a = 2

                """loss to torch.Tenser. Stąd (kozacka) metoda .backward()"""
                loss.backward()
                """tensor (loss) liczy TYLKO 'grad'.  A przecież nam zależy na minimum (w końcu SDG)"""
                optimizer.step()

                """loss w tym momencie to Tensor(1,1,1). Tensory tej kategorii mają metodę .item(), która zwraca ich wartość."""
                """+= -> bo robimy <stochastic> GD (średnie, sumy, itd.)"""
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    # print(running_loss)
                    # sys.exit()
                    running_loss = 0.0
            correct = 0
            total = 0
            """torch.no_grad() -> cntxmngr, który zapewnia, że nic nie odpali .backward() (takie zabezpieczenie)"""
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    outputs = net(images)
                    # TODO to trzeba bedzie ekstrahowac
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    """do 'correct' dodajemy 0 lub 1, jednak jest to zapisany w nieintuicyjny sposób [sum() == WTF]"""
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, 100 * correct / total))


def main():
    torch.manual_seed(0)
    transform0 = transforms.Compose([transforms.ToTensor()])

    # return

    trainset, testset = [torchvision.datasets.MNIST(root=rf"C:\Datasets", download=True, train=b, transform=transform0) for b in [True, False]]
    # trainset, testset = [ShapesDataset(rf"C:\Datasets\mnist_", slice_=s) for s in [slice(0, 60000), slice(60000, 70000)]]
    net_base = Net()
    trainer = MnistTrainer(net=net_base, datasets=(trainset, testset), no_epoch=3)
    trainer.train()

"""
[1,   100] loss: 0.679
[1,   200] loss: 0.282
[1,   300] loss: 0.215
[1,   400] loss: 0.192
Accuracy of the network on the 10000 test images: 95.01 %
[2,   100] loss: 0.143
[2,   200] loss: 0.136
[2,   300] loss: 0.135
[2,   400] loss: 0.127
Accuracy of the network on the 10000 test images: 96.09 %
"""


if __name__ == '__main__':
    main()




"""SCRATCH"""
# labels[0]
# Out[5]: tensor(6)
# labels[0].shape
# Out[6]: torch.Size([])

# outputs[0]
# Out[7]:
# tensor([47.0333, -8.8092, -31.7119, -15.8912, -14.0070, 58.2908, 15.3053,
#         16.3086, 14.9250, 1.6518], grad_fn= < SelectBackward >)
# outputs[0].shape
# Out[8]: torch.Size([10])

# outputs[0:2]
# Out[2]:
# tensor([[-0.0017,  1.1775, -0.3094, -0.3539,  0.0291, -0.0736, -0.1235,  0.0097, -0.9908,  0.5361],
#         [ 0.1747,  0.5950,  0.0957, -0.3287, -0.1601, -0.0745,  0.2418,  0.0014, -0.3589,  0.3077]], grad_fn=<SliceBackward>)
# labels[0:2]
# Out[3]: tensor([2, 5])

# [1,   100] loss: 1.417
# [1,   200] loss: 0.529
# [1,   300] loss: 0.405
# [1,   400] loss: nan

# TODO zbierzmy logi co tu sie odjebalo
# TODO zamiana sum na mean()
# TODO zrob jakies aweraging
# TODO co zrobic z zerem ???

# def multiindex_nll_loss(outputs, labels):
#     loss = -torch.sum(torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels))
#     return loss

"""syntetyk"""
# outputs = torch.Tensor([[0.05, 0.9, 0.3, 0.05, 0.05, 0.05], [0.01, 0.95, 0.05, 0.3, 0.05, 0.05]]) #torch.Size([2, 6])
# labels = torch.Tensor([[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]]) #torch.Size([2, 6])


"""natural (stary init)"""
# labels = torch.tensor([6, 2], dtype=torch.long) # torch.Size([2])
# outputs = torch.Tensor([[47.0333, -8.8092, -31.7119, -15.8912, -14.0070, 58.2908, 15.3053, 16.3086, 14.9250, 1.6518], [ 19.9257,  -6.3817, -14.2751,  21.4164, -21.7247,  49.5124,  -2.5922, -21.6784,  -1.3972,  10.1371]]) #torch.Size([2, 10])


"""natural (nowy init)"""
# labels = torch.tensor([2, 5], dtype=torch.long) # torch.Size([2])
# outputs = torch.Tensor([[-0.0017,  1.1775, -0.3094, -0.3539,  0.0291, -0.0736, -0.1235,  0.0097, -0.9908,  0.5361], [ 0.1747,  0.5950,  0.0957, -0.3287, -0.1601, -0.0745,  0.2418,  0.0014, -0.3589,  0.3077]]) #torch.Size([2, 10])


"""skurwialy case 389"""
labels = torch.tensor([5], dtype=torch.long) # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
outputs = torch.Tensor([[-18.5508, -17.9211, -19.8926, -13.2589, -16.2123,  16.8303, -14.1339, -20.7289, -11.2858, -14.3649]]) #torch.Size([2, 10])


labels1 = F.one_hot(labels, 10) # torch.Size([2, 10])
outputs1 = 0.9999 * torch.sigmoid(outputs)
multiindex_nll_loss(outputs1, labels1)

outputs = outputs1
labels = labels1

torch.log(1 - torch.sigmoid(torch.tensor([47.]))) # TODO <- to jest problem

# TODO chyba trzeba zaepsilonic (1-epsilon)
# TODO wrzucic ten sigmoid wczensiej
# TODO sprawdzic, jak sie wagi zmieniaja

# TODO test na prodzie
# TODO target_transform
# TODO moze na slacku pisze, co to za funkcja


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



