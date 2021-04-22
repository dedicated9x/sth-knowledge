import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import sys


def truncated_normal_(tensor, mean=0, std=1, generator_=None):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_(generator=generator_)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class Linear(torch.nn.Module):
    generator = None

    """in_features, out_features -> <liczba> neuronów przed, <liczba> neuronów po tej warstwie."""
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.5, generator_=Linear.generator)
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


class MnistTrainer(object):
    def __init__(self, net, no_epoch=20):
        self.net = net
        self.no_epoch = no_epoch
        transform = transforms.Compose(
                [transforms.ToTensor()])
        self.trainset = torchvision.datasets.MNIST(
            root=rf"C:\Datasets",
            download=True,
            train=True,
            transform=transform)
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
    Linear.generator = torch.Generator().manual_seed(1234567890)
    net_base = Net()
    trainer = MnistTrainer(net=net_base, no_epoch=1)
    trainer.train()


"""
[1,   100] loss: 2.033
[1,   200] loss: 0.553
[1,   300] loss: 0.427
[1,   400] loss: 0.394
Accuracy of the network on the 10000 test images: 90.56 %
"""


# TODO teraz pewnie szuffle
    # TODO ustawic seeda i zrobic zapis

# TODO wczytaj, jak dataset, a nie jak mnista

if __name__ == '__main__':
    main()


