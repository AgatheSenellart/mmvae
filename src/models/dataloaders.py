# Dataloaders classes to be used with any model

from torch.utils.data import DataLoader
import os
import torch
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision import datasets, transforms

class MNIST_DL():
    def __init__(self, data_path, type):
        self.type = type
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True,device='cuda'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        datasetC = datasets.MNIST if self.type == 'numbers' else datasets.FashionMNIST
        train = DataLoader(datasetC('../data', train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasetC('../data', train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test


class MNIST_FASHION_DATALOADER():

    def __init__(self, data_path):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        print(self.data_path)
        if not (os.path.exists(self.data_path + 'train-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + 'train-ms-fashion-idx.pt')
                and os.path.exists(self.data_path + 'test-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + 'test-ms-fashion-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + 'train-ms-mnist-idx.pt')
        t_fashion = torch.load(self.data_path + 'train-ms-fashion-idx.pt')
        s_mnist = torch.load(self.data_path + 'test-ms-mnist-idx.pt')
        s_fashion = torch.load(self.data_path + 'test-ms-fashion-idx.pt')

        # load base datasets
        t1,s1 = MNIST_DL(self.data_path,'numbers').getDataLoaders(batch_size,shuffle,device)
        t2,s2 = MNIST_DL(self.data_path,'fashion').getDataLoaders(batch_size,shuffle,device)


        train_mnist_fashion = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_fashion[i], size=len(t_fashion))
        ])

        test_mnist_fashion = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_fashion[i], size=len(s_fashion))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_fashion, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_fashion, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test