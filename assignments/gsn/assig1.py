import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision
import pandas as pd
import numpy as np
import pathlib as pl
import itertools
import sys
import seaborn as sn


import matplotlib.pyplot as plt


class Linear(torch.nn.Module):
    """This class has been """
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        r = x.matmul(self.weight.t()) # .t() => wyciąga z obiektu Parameter, jego obiekt bazowy Tensor (obliczenia tego wymagają)
        r += self.bias
        return r

class Net(nn.Module):
    def __init__(self, dense_first, dense_core, dense_last, nonlin_outlayer=None, conv=None):
        super(Net, self).__init__()
        self.conv = conv if conv is not None else torch.nn.Identity()
        self.dense_first = dense_first
        self.dense_core = dense_core
        self.dense_last = dense_last
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

    @staticmethod
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, Linear):
            m.reset_parameters()

    def reset_parameters(self):
        self.conv.apply(Net.weight_reset)
        self.dense_first.apply(Net.weight_reset)
        self.dense_core.apply(Net.weight_reset)
        self.dense_last.apply(Net.weight_reset)

    def with_parts(self, **kwargs):
        # new_net = Net(dense_first=self.dense_first, dense_core=self.dense_core, dense_last=self.dense_last, nonlin_outlayer=self.nonlin_outlayer, conv=self.conv)
        new_net = Net(**self.get_parts())
        for k, v in kwargs.items():
            setattr(new_net, k, v)
        return new_net

    def get_parts(self):
        return {'dense_first': self.dense_first, 'dense_core': self.dense_core, 'dense_last': self.dense_last, 'nonlin_outlayer': self.nonlin_outlayer, 'conv': self.conv}

    def to_device(self, device):
        self.conv.to(device)
        self.dense_first.to(device)
        self.dense_core.to(device)
        self.dense_last.to(device)


class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, slice_=None, augmented=False, transform=None, target_transform=None):
        self.img_dir = pl.Path(root).joinpath('data')
        self.df = pd.read_csv(self.img_dir.joinpath('labels.csv'))
        if slice_ is not None:
            self.df = self.df[slice_]
        self.images = torch.stack([torchvision.io.read_image(self.img_dir.joinpath(name).__str__())[0:1] / 255 for name in self.df['name']])
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
    def __init__(self, datasets, device, loss, acc, logger):
        self.trainset, self.testset = datasets
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False)
        self.loss = loss
        self.accuracy = acc
        self.device = device
        self.logger = logger
        self.print_period = 20

    def train(self, net, name, no_epoch=20, verbose=0, reset=True):
        """FOCUS: sgd dostaje info o sieci, jaką będzie trenował"""
        if reset:
            net.reset_parameters()
        net.to_device(self.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
        log_acc = []

        for epoch in range(no_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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
                    if verbose == 1:
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / self.print_period))
                    running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.testloader:
                    inputs_, labels_ = data
                    inputs_, labels_ = inputs_.to(self.device), labels_.to(self.device)
                    outputs_ = net(inputs_)
                    outputs_transformed, labels_transformed = self.accuracy(outputs_, labels_)
                    correct += self.count_matches(outputs_transformed, labels_transformed)
                    total += outputs_.shape[0]
            accuracy = 100 * correct / total
            log_acc.append(accuracy)
            if verbose == 1:
                print('Accuracy of the network on the {} test images: {} %'.format(total, accuracy))
        # return log_acc
        self.logger.save(name, log_acc)
        return None

    def count_matches(self, outputs_transformed, labels_transformed):
        """outputs & labels MUST be integer tensors"""
        no_correct = (outputs_transformed == labels_transformed).all(dim=1).int().sum().item()
        return no_correct

class CustomFunctional:
    counts2class = None # Will be calculated in external scope
    device = None

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
        js = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).repeat(6).to(device)
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
        labels_ = torch.tensor([CustomFunctional.counts2class[tuple(e.cpu().numpy())] for e in labels]).to(device)
        loss = CustomFunctional.loss_nll(outputs, F.one_hot(labels_, 135))
        return loss

    @ classmethod
    def acctransform_count135(cls, outputs, labels):
        outputs_transformed = torch.topk(outputs, 1).indices
        labels_ = torch.tensor([CustomFunctional.counts2class[tuple(e.cpu().numpy())] for e in labels]).to(device)
        labels_transformed = labels_.unsqueeze(dim=1)
        return outputs_transformed, labels_transformed

class Logger:
    def __init__(self):
        self.names2logs = {}

    def save(self, name, log):
        self.names2logs[name] = log

    def acc(self, name):
        return max(logger.names2logs[name])

    def show_compared(self, names):
        names2logs_subdict = {k: v for k, v in self.names2logs.items() if k in names}
        Utils.plot_logs(names2logs_subdict)


class Utils:
    @staticmethod
    def get_loss_inputs(trainset, net, mb_size):
        loader = torch.utils.data.DataLoader(trainset, batch_size=mb_size, shuffle=False)
        inputs, labels = next(loader.__iter__())
        outputs = net(inputs)
        return outputs, labels

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

    @staticmethod
    def plot_logs(names2logs: dict):
        fig, ax = plt.subplots(1, 1)
        for label_, log in names2logs.items():
            ax.plot(range(len(log)), log, marker='o', label=label_)
        ax.legend(loc="lower right")
        ax.hlines(100, 0, max([len(l) for l in names2logs.values()]) - 1, color='black')


REF = {}
logger = Logger()
device = 'cpu'

def main():
    # !wget https://www.mimuw.edu.pl/~ciebie/gsn-2021-1.zip
    # !unzip gsn-2021-1.zip -d gsn-2021-1

    if 'google.colab' in str(get_ipython()):
        path_to_data = rf"gsn-2021-1"
    else:
        torch.manual_seed(0)
        path_to_data = rf"C:\Datasets\gsn-2021-1"

    CF = CustomFunctional
    CF.init()

    CF.device = device

    
    testset = ShapesDataset(path_to_data, slice(9000, 10000))
    trainset = ShapesDataset(path_to_data, slice(0, 9000))
    # trainset = ShapesDataset(rf"C:\Datasets\gsn-2021-1", slice(0, 9000), augmented=True)

    conv_arbitrary = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2), nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    mnistslayer_head = nn.Sequential(Linear(784, 64), nn.ReLU())
    mnistslayer_body = nn.Sequential(Linear(64, 64), nn.ReLU())
    dlast6 = Linear(64, 6)
    dlast60 = Linear(64, 60)
    dlast135 = Linear(64, 135)


    net_trivial6 = Net(dense_first=mnistslayer_head, dense_core=mnistslayer_body, dense_last=dlast6, nonlin_outlayer=torch.sigmoid)
    net_base6 = net_trivial6.with_parts(conv=conv_arbitrary)
    net_base60 = net_base6.with_parts(dense_last=dlast60, nonlin_outlayer=CF._10_piecewise_softmax)
    net_base135 = net_base6.with_parts(dense_last=dlast135, nonlin_outlayer=lambda outputs: torch.softmax(outputs, dim=1))

    # net_base6.load_state_dict(torch.load(rf"C:\temp\output\state2.pickle"))


    trainer_classify6 = MnistTrainer(datasets=(trainset, testset), device=device, logger=logger, loss=CF.loss_classify6, acc=CF.acctransform_classify6)
    trainer_count60 = MnistTrainer(datasets=(trainset, testset), device=device, logger=logger, loss=CustomFunctional.loss_count60, acc=CustomFunctional.acctransform_count60)
    trainer_count135 = MnistTrainer(datasets=(trainset, testset), device=device, logger=logger, loss=CustomFunctional.loss_count135, acc=CustomFunctional.acctransform_count135)


    trainer_classify6.train(net=net_base6, name='name1', no_epoch=2, verbose=1)
    trainer_count60.train(net=net_base60, name='name2', no_epoch=2, verbose=1)
    trainer_count135.train(net=net_base135, name='name3', no_epoch=20, verbose=1, reset=True)

    logger.show_compared(['name1', 'name3'])

    # torch.save(net_base6.state_dict(), rf"C:\temp\output\state2.pickle")

"""
[1,    20] loss: 4.089
[1,    40] loss: 3.819
[1,    60] loss: 3.806
Accuracy of the network on the 1000 test images: 8.7 %
[2,    20] loss: 3.945
[2,    40] loss: 3.745
[2,    60] loss: 3.363
Accuracy of the network on the 1000 test images: 18.9 %
[1,    20] loss: 55.979
[1,    40] loss: 47.789
[1,    60] loss: 47.710
Accuracy of the network on the 1000 test images: 0.0 %
[2,    20] loss: 50.206
[2,    40] loss: 47.713
[2,    60] loss: 47.529
Accuracy of the network on the 1000 test images: 0.0 %
[1,    20] loss: 6.186
[1,    40] loss: 5.849
[1,    60] loss: 5.794
Accuracy of the network on the 1000 test images: 1.8 %
[2,    20] loss: 6.000
[2,    40] loss: 5.697
[2,    60] loss: 5.695
Accuracy of the network on the 1000 test images: 0.6 %
"""

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
if __name__ == '__main__':
    main()


"""CONFUSION MATRIX"""
# testset = REF['SET']
# net = REF['NET']
#
# # z1 = net(testset.images)
#
# labels2classids = {
#     (1, 1, 0, 0, 0, 0): 0,
#     (1, 0, 1, 0, 0, 0): 1,
#     (1, 0, 0, 1, 0, 0): 2,
#     (1, 0, 0, 0, 1, 0): 3,
#     (1, 0, 0, 0, 0, 1): 4,
#     (0, 1, 1, 0, 0, 0): 5,
#     (0, 1, 0, 1, 0, 0): 6,
#     (0, 1, 0, 0, 1, 0): 7,
#     (0, 1, 0, 0, 0, 1): 8,
#     (0, 0, 1, 1, 0, 0): 9,
#     (0, 0, 1, 0, 1, 0): 10,
#     (0, 0, 1, 0, 0, 1): 11,
#     (0, 0, 0, 1, 1, 0): 12,
#     (0, 0, 0, 1, 0, 1): 13,
#     (0, 0, 0, 0, 1, 1): 14,
# }
#
#
# outputs_, labels_ = CustomFunctional.acctransform_classify6(net(testset.images), testset.labels)
# cm = np.zeros(shape=(15, 15))
# for o, l in zip(outputs_, labels_):
#     predicted_id = labels2classids[tuple(o.numpy())]
#     true_id = labels2classids[tuple(l.numpy())]
#     cm[true_id, predicted_id] += 1
#
# symbols = '□○△▷▽◁'
# label2symbol = lambda x: ''.join([symbols[idx] for idx, bool_ in enumerate(x) if bool_ == 1])
# class_symbols = [label2symbol(k) for k in labels2classids.keys()]
# df_cm = pd.DataFrame(cm, index=class_symbols, columns=class_symbols)
# sn.heatmap(df_cm, annot=True)
#
# """Conclusion"""
# pd.Series([(o == l).int().sum().item() for o, l in zip(outputs_, labels_)]).replace({6: 'all matched', 4: '1 matched', 2: 'none matched'}).value_counts()
#
# """One shape"""
# indices_of_ones = lambda x: (x == 1).nonzero(as_tuple=True)[0].numpy().tolist()
# paired_indices = lambda x, y: [(i, i) for i in set(x) & set(y)] + [(i, j) for i, j in zip(list(set(x) - set(y)), list(set(y) - set(x)))]
# [(indices_of_ones(o), indices_of_ones(l)) for o, l in zip(outputs_[:10], labels_[:10])]
# [paired_indices(indices_of_ones(o), indices_of_ones(l)) for o, l in zip(outputs_[:10], labels_[:10])]
# cm = np.zeros(shape=(6, 6))
# for o, l in zip(outputs_, labels_):
#     for predicted_id, true_id in paired_indices(indices_of_ones(o), indices_of_ones(l)):
#         cm[true_id, predicted_id] += 1
#
# class_symbols = list(symbols)
# df_cm = pd.DataFrame(cm, index=class_symbols, columns=class_symbols)
# sn.heatmap(df_cm, annot=True, fmt='g')


