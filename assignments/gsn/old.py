


"""plot logs"""
# log1 = [6.8, 16.3, 17.2, 25.0, 30.8, 54.0, 67.3, 73.8, 73.7, 74.5, 78.2, 83.9, 76.8, 82.4, 81.6]
# log2 = [1.2, 1.1, 1.6, 3.3, 9.4, 15.5, 23.3, 22.8, 25.5, 27.7, 29.3, 25.9, 26.4, 29.9, 27.2, 27.1, 27.6, 26.2, 26.0, 26.2]
# logs = [(log1, 'conv2'), (log2, 'dense')]
# import matplotlib.pyplot as plt
# Utils.plot_logs(logs)


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


"""porownanie datasetow"""
# ts1 = MnistTrainDataset(rf"C:\Datasets\mnist_train", slice_=slice(20, 30))
# tup1 = ts1.__getitem__(2)
# z1 = tup1[0].numpy()[0, :, :]


"""polaczenie dataframow"""
# f1 = lambda x: x[:4] + '6' +  x[5:]
# df_train = pd.read_csv(rf"C:\Datasets\mnist_train\data\labels.csv")
# df_test = pd.read_csv(rf"C:\Datasets\mnist_test\data\labels.csv")
# df_test['name'] = df_test['name'].apply(f1)
# combined = df_train.append(df_test, ignore_index=True)
# combined.to_csv(rf"C:\temp\gsn_assig_1_output\labels.csv", index=False)

"""zmiana nazw testowych"""
# f1 = lambda x: x[:4] + '6' +  x[5:]
# x = 'img_00002.png'
# root = pl.Path(rf"C:\temp\src\data")
# images = list(root.glob('**/*'))
# [i.rename(i.with_name(f1(i.name))) for i in images]

"""porownanie datasetow"""
# transform = transforms.Compose([transforms.ToTensor()])
# ts0 = torchvision.datasets.MNIST(root=rf"C:\Datasets", download=True, train=True, transform=transform)
# ts1 = MnistTrainDataset(rf"C:\Datasets\mnist_train")
# tup0 = ts0.__getitem__(2)
# tup1 = ts1.__getitem__(2)
# z0 = tup0[0].numpy()[0, :, :]
# z1 = tup1[0].numpy()[0, :, :]

"""csv train"""
# trainer = MnistTrainer(net=Net(), no_epoch=1)
# ser_names = [e.name for e in pl.Path(rf"C:\Datasets\mnist_bad").glob('**/*')]
# ser_labels = trainer.trainset.test_labels.numpy()
# df = pd.DataFrame().assign(name=ser_names, label=ser_labels)
# df.to_csv(rf"C:\temp\gsn_assig_1_output\label.csv", index=False)

"""csv test"""
# trainer = MnistTrainer(net=Net(), no_epoch=1)
# ser_names = [e.name for e in pl.Path(rf"C:\Datasets\mnist_test\data").glob('**/*')]
# ser_labels = trainer.testset.test_labels.numpy()
# df = pd.DataFrame().assign(name=ser_names, label=ser_labels)
# df.to_csv(rf"C:\temp\gsn_assig_1_output\label.csv", index=False)

"""images train"""
# for i in range(60000):
#     arr = trainer.trainset.train_data[i, :, :].numpy()
#     # arr = simpler(arr)
#     filename = f'img_{str(i).zfill(5)}.png'
#     cv2.imwrite(path_root.joinpath(filename).__str__(), arr)
#
#     print(i)

"""images test"""
# import cv2
# path_root = pl.Path(rf"C:\temp\gsn_assig_1_output")
# trainer = MnistTrainer(net=Net(), no_epoch=1)
# for i in range(10000):
#     arr = trainer.testset.train_data[i, :, :].numpy()
#     filename = f'img_{str(i).zfill(5)}.png'
#     cv2.imwrite(path_root.joinpath(filename).__str__(), arr)

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
