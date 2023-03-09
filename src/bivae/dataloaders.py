# Dataloaders classes to be used with any model

from torch.utils.data import DataLoader
import os
import torch
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np


from .datasets import CelebA
from torchvision.transforms import ToTensor

########################################################################################################################
########################################## DATASETS ####################################################################


class BasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform = None):

        self.data = data # shape len_data x ch x w x h
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 0 # We return 0 as label to have a dataset that is homogeneous with the other datasets

class MultimodalBasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None, length=None):
        # data of shape n_mods x len_data x ch x w x h
        self.lenght = len(data[0]) if length is None else length
        self.datasets = [BasicDataset(d, transform) for d in data ]

    def __len__(self):
        return self.lenght

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)



##########################################################################################################
#################################### UNIMODAL DATALOADERS ################################################


class MNIST_DL():
    def __init__(self, data_path, type):
        self.type = type
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True,device='cuda', transform=None):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == "cuda" else {}
        if transform is None:
            tx = transforms.ToTensor()
        else :
            tx = transform
        datasetC = datasets.MNIST if self.type == 'numbers' else datasets.FashionMNIST
        train = DataLoader(datasetC(self.data_path , train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasetC(self.data_path, train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

class SVHN_DL():

    def __init__(self, data_path = '../data'):
        self.data_path = data_path
        return

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

        train = DataLoader(datasets.SVHN(self.data_path, split='train', download=True, transform=transform),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN(self.data_path , split='test', download=True, transform=transform),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test




########################################################################################################################
####################################### MULTIMODAL DATALOADERS #########################################################




class MNIST_SVHN_DL():

    def __init__(self, data_path='../data'):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor(), len_train=None):

        if not (os.path.exists(self.data_path + '/train-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + '/train-ms-svhn-idx.pt')
                and os.path.exists(self.data_path + '/test-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + '/test-ms-svhn-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + '/train-ms-mnist-idx.pt')
        t_svhn = torch.load(self.data_path + '/train-ms-svhn-idx.pt')
        s_mnist = torch.load(self.data_path + '/test-ms-mnist-idx.pt')
        s_svhn = torch.load(self.data_path + '/test-ms-svhn-idx.pt')

        # load base datasets
        t1, s1 = MNIST_DL(self.data_path, type='numbers').getDataLoaders(batch_size, shuffle, device, transform)
        t2, s2 = SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        
        # shuffle to be able to reduce size of the dataset
        
        rd_idx = np.random.RandomState(seed=42).permutation(len(t_mnist))
        t_mnist, t_svhn = t_mnist[rd_idx], t_svhn[rd_idx]
        if len_train is None: 
            len_train = len(t_mnist)
        
        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len_train),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len_train)
        ])
        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        # Split between test and validation while fixing the seed to ensure that we always have the same sets
        len_val = min(10000, len(train_mnist_svhn)//10)
        train_set, val_set = random_split(train_mnist_svhn,
                                         [len(train_mnist_svhn)-len_val,
                                          len_val],
                                         generator=torch.Generator().manual_seed(42))



        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val
    


class MNIST_SVHN_FASHION_DL():

    def __init__(self, data_path='../data'):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):

        if not (os.path.exists(self.data_path + '/train-msf-mnist-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + '/train-msf-mnist-idx.pt')
        t_svhn = torch.load(self.data_path + '/train-msf-svhn-idx.pt')
        t_fashion = torch.load(self.data_path + '/train-msf-fashion-idx.pt')
        s_mnist = torch.load(self.data_path + '/test-msf-mnist-idx.pt')
        s_svhn = torch.load(self.data_path + '/test-msf-svhn-idx.pt')
        s_fashion = torch.load(self.data_path + '/test-msf-fashion-idx.pt')

        # load base datasets
        t1, s1 = MNIST_DL(self.data_path, type='numbers').getDataLoaders(batch_size, shuffle, device, transform)
        t2, s2 = SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        t3, s3 = MNIST_DL(self.data_path, type='fashion').getDataLoaders(batch_size, shuffle,device, transform)

        # shuffle to be able to reduce size of the dataset
        rd_idx = np.random.permutation(len(t_mnist))
        t_mnist, t_svhn, t_fashion = t_mnist[rd_idx], t_svhn[rd_idx], t_fashion[rd_idx]
        
        rd_idx_test = np.random.permutation(len(s_mnist))
        s_mnist, s_svhn, s_fashion = s_mnist[rd_idx_test], s_svhn[rd_idx_test], s_fashion[rd_idx_test]
        
        # len_train = 100000
        len_train = len(t_mnist)
        
        # len_test = 1000
        len_test = len(s_mnist)
        
        train_msf = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len_train),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len_train),
            ResampleDataset(t3.dataset, lambda d,i : t_fashion[i], size=len_train)
        ])
        
        test_msf = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len_test),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len_test),
            ResampleDataset(s3.dataset, lambda d,i : s_fashion[i], size=len_test)
        ])

        # Split between test and validation while fixing the seed to ensure that we always have the same sets
        train_set, val_set = random_split(train_msf,
                                         [len(train_msf)-10000,
                                          10000],
                                         generator=torch.Generator().manual_seed(42))



        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_msf, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val



class CELEBA_DL():

    def __init__(self, data_path='../data/'):
        self.data_path = data_path 
    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', len_train=None, transform=ToTensor()):

        train_dataset = CelebA(self.data_path, 'train', transform=transform, len=len_train)
        test = CelebA(self.data_path, 'test', transform=transform)
        val = CelebA(self.data_path, 'val', transform=transform)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}

        train_dataloader = DataLoader(train_dataset, batch_size, shuffle, **kwargs)
        test_dataloader = DataLoader(test, batch_size, shuffle=False, **kwargs)
        val_dataloader = DataLoader(val, batch_size,shuffle=False,**kwargs )
        return train_dataloader, test_dataloader, val_dataloader
