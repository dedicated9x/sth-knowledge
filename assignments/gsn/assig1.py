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
        truncated_normal_(self.weight, std=0.5)
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


class MnistTrainDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.img_dir = pl.Path(root).joinpath('data')
        img_labels = pd.read_csv(self.img_dir.joinpath('labels.csv'))
        self.images = [read_image(self.img_dir.joinpath(name).__str__()) / 255 for name in img_labels['name']]
        self.labels = img_labels['label']
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


class MnistTrainer(object):
    def __init__(self, net, no_epoch=20):
        self.net = net
        self.no_epoch = no_epoch
        transform = transforms.Compose(
                [transforms.ToTensor()])
        self.trainset = torchvision.datasets.MNIST(root=rf"C:\Datasets", download=True, train=True, transform=transform)
        # self.trainset = MnistTrainDataset(rf"C:\Datasets\mnist_train")
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=MB_SIZE, shuffle=True, num_workers=4)

        self.testset = torchvision.datasets.MNIST(
            root=rf"C:\Datasets",
            train=False,
            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=4)

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
                loss = criterion(outputs, labels)
                """loss to torch.Tenser. Stąd (kozacka) metoda .backward()"""
                loss.backward()
                """tensor liczy TYLKO 'grad'.  A przecież nam zależy na minimum (w końcu SDG)"""
                optimizer.step()

                """loss w tym momencie to Tensor(1,1,1). Tensory tej kategorii mają metodę .item(), która zwraca ich wartość."""
                """+= -> bo robimy <stochastic> GD (średnie, sumy, itd.)"""
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            correct = 0
            total = 0
            """torch.no_grad() -> cntxmngr, który zapewnia, że nic nie odpali .backward() (takie zabezpieczenie)"""
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    """do 'correct' dodajemy 0 lub 1, jednak jest to zapisany w nieintuicyjny sposób [sum() == WTF]"""
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, 100 * correct / total))


def main():
    # Linear.generator = torch.Generator().manual_seed(1234567890)
    torch.manual_seed(0)
    net_base = Net()
    trainer = MnistTrainer(net=net_base, no_epoch=1)
    trainer.train()

"""
[1,   100] loss: 2.304
[1,   200] loss: 0.618
[1,   300] loss: 0.492
[1,   400] loss: 0.416
Accuracy of the network on the 10000 test images: 90.11 %
[2,   100] loss: 0.337
[2,   200] loss: 0.321
[2,   300] loss: 0.317
[2,   400] loss: 0.293
Accuracy of the network on the 10000 test images: 91.72 %
"""


if __name__ == '__main__':
    main()




"""SCRATCH"""

# self = MnistTrainDataset(rf"C:\Datasets\mnist_train")


# transform = transforms.Compose([transforms.ToTensor()])
# ts0 = torchvision.datasets.MNIST(root=rf"C:\Datasets", download=True, train=True, transform=transform)
# ts1 = MnistTrainDataset(rf"C:\Datasets\mnist_train")
# tup0 = ts0.__getitem__(2)
# tup1 = ts1.__getitem__(2)
# z0 = tup0[0].numpy()[0, :, :]
# z1 = tup1[0].numpy()[0, :, :]


# TODO poprawic annotacje dla
# TODO zrobic druga wersje dla testowego
# TODO spakowac to do jednego


"""csv train"""
# trainer = MnistTrainer(net=Net(), no_epoch=1)
# ser_names = [e.name for e in pl.Path(rf"C:\Datasets\mnist_bad").glob('**/*')]
# ser_labels = trainer.trainset.test_labels.numpy()
# df = pd.DataFrame().assign(name=ser_names, label=ser_labels)
# df.to_csv(rf"C:\temp\gsn_assig_1_output\label.csv", index=False)

"""60000 zapis"""
# for i in range(60000):
#     arr = trainer.trainset.train_data[i, :, :].numpy()
#     # arr = simpler(arr)
#     filename = f'img_{str(i).zfill(5)}.png'
#     cv2.imwrite(path_root.joinpath(filename).__str__(), arr)
#
#     print(i)


"""MNIST vs GSN - testy"""
# mnist_out = trainer.trainset.train_data[0, :, :].numpy()
# mnist_out = np.vectorize(simpler)(mnist_out)
# cv2.imwrite(path_root.joinpath('mnist.png').__str__(), mnist_out)
#
# ex_gsn = cv2.imread(path_root.joinpath('gsn.png').__str__()) # bit depth = 8
# ex_mnist = cv2.imread(path_root.joinpath('mnist.png').__str__()) # bit depth = 32
#
# gsn0 = ex_gsn[:, :, 0]
# gsn1 = ex_gsn[:, :, 1]
# gsn2 = ex_gsn[:, :, 2]
#
# mnist0 = ex_mnist[:, :, 0]
# mnist1 = ex_mnist[:, :, 1]
# mnist2 = ex_mnist[:, :, 2]
#
# summary(gsn0)
# summary(mnist0)
# summary(mnist_out)

