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
