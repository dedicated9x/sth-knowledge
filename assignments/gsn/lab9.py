import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torchvision
import torchvision.transforms as transforms

''''
Tasks:
1. Check that the given implementation reaches 95% test accuracy for
   architecture input-64-64-10 in a few thousand batches.

2. Improve initialization and check that the network learns much faster
   and reaches over 97% test accuracy.

3. Check, that with proper initialization we can train architecture
   input-64-64-64-64-64-10, while with bad initialization it does
   not even get off the ground.

4. Add dropout implemented in pytorch

5. Check that with 10 hidden layers (64 units each) even with proper
    initialization the network has a hard time to start learning.

6. Implement batch normalization (use train mode also for testing
       - it should perform well enough):
    * compute batch mean and variance
    * add new variables beta and gamma
    * check that the networks learns much faster for 5 layers
    * check that the network learns even for 10 hidden layers.

Bonus task:

Design and implement in pytorch (by using pytorch functions)
   a simple convnet and achieve 99% test accuracy.

Note:
This is an exemplary exercise. MNIST dataset is very simple and we are using
it here to get resuts quickly.
To get more meaningful experience with training convnets use the CIFAR dataset.
'''

"""to po prostu inicjalizuje wagi z rokładu normalnego (jakiegoś truncated)"""
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

"""'Linear', bo przecież, jak dane przechodzą przez sieć, to jest to zwykłe przekształcenie liniowe"""
class Linear(torch.nn.Module):
    """in_features, out_features -> <liczba> neuronów przed, <liczba> neuronów po tej warstwie."""
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        """'Parameter' to 'Tensor', który zauważa, że jest atrybutem dla 'Module' i wpisuje się mu w odpowiednie miejsca przy inicjalizacji."""
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    """User-defined funckją używana <tylko> przy inicjalizacji warstwy (do zainicjalizowania wartości wag)."""
    def reset_parameters(self):
        # init.kaiming_normal_(self.weight, mode='fan_in') # dobra inicjalizacja dla Relu (95->97) # ma znaczene przy wielu warstawch
        # init.xavier_normal_(self.w) -> druga opcja
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
        """
        Linear to sieć/warstwa zaimplementowa powyżej.
        Ad. 'sieć/warstwa' w ogólności sieci i warstwy mają taki sam interfejs (input + output). Stąd zapewne implementacja, w której dziedziczymy z tej samej klasy.
        """
        self.fc1 = Linear(784, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 10)

        # self.dropout = nn.Dropout(0.25)

    """x -> inputy (wchodzą w petli SGD)"""
    def forward(self, x):
        """.view() to .reshape() dla tensorów"""
        x = x.view(-1, 28 * 28)
        # x = self.dropout(x)
        """nn.Model.__call__() odpala .forward()"""
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


MB_SIZE = 128


class MnistTrainer(object):
    def __init__(self):
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
        net = Net()
        """criterion -> zwykła funkcja, której użyjemy później"""
        criterion = nn.CrossEntropyLoss()
        """FOCUS: sgd dostaje info o sieci, jaką będzie trenował"""
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

        for epoch in range(20):
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
                """w innych interwałach pokazujemy <loss>, a w innych <accuracy> (te drugie jest dalej) """
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            correct = 0
            total = 0
            """torch.no_grad() -> cntxmngr, który zapewnia, że nic nie odpali .backward() (takie zabezpieczenie)"""
            with torch.no_grad():
                #WPISAC GDZIE INDZIEJ W KODZIE: torch.is_grad_enabled() # sprawdza, czy jestesmy w torch.no_grad


                # net.eval() <- np. dla batch normalizacji
                # Ta funkcja sprawia, że cała sieć przechodzi w tryb ewaluacji (zamist teeningowe)
                # np. wyłączy dropouta.
                for data in self.testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    """do 'correct' dodajemy 0 lub 1, jednak jest to zapisany w nieintuicyjny sposób [sum() == WTF]"""
                    correct += (predicted == labels).sum().item()
                # net.train() -> wracamy do trybu treningowego

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, 100 * correct / total))


def main():
    trainer = MnistTrainer()
    trainer.train()

# TODO sprawdzic, czy jest 95%
# TODO zrobic kopie
# TODO ustawic seeda i zrobic zapis

# TODO wczytaj, jak dataset, a nie jak mnista

if __name__ == '__main__':
    main()
