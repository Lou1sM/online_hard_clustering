import torch
from os.path import join
import torchvision
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from pdb import set_trace
import numpy as np


class CifarLikeDataset(data.Dataset):
    def __init__(self,x,y):
        self.data, self.targets = x,y
        assert len(x) == len(y)
    def __len__(self): return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx], self.targets[idx]

def get_cifar10(test):
    transform = Compose([ToTensor(),Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)
    if test==2:
        trainset = testset
        trainset.data = trainset.data[:1000]
        trainset.targets = trainset.targets[:1000]
    elif test==1:
        trainset = testset
        rand_idxs = torch.randint(len(trainset),size=(10000,))
        trainset.data = trainset.data[rand_idxs]
        trainset.targets = torch.tensor(trainset.targets)[rand_idxs].tolist()
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

    trainset.data = np.concatenate([trainset.data,testset.data])
    trainset.targets = np.concatenate([trainset.targets,testset.targets])
    return trainset


def get_imagenet_tiny(test):
    data_for_each_class = []
    labels_for_each_class = []
    for class_idx in range(200):
        np_class_data = np.load(join('tiny-imagenet-200/np_data',f'{class_idx}.npy'))
        np_class_labels = np.load(join('tiny-imagenet-200/np_data',f'{class_idx}_labels.npy'))
        if test == 1:
            np_class_data = np_class_data[:50]
            np_class_labels = np_class_labels[:50]
        elif test == 2:
            np_class_data = np_class_data[:20]
            np_class_labels = np_class_labels[:20]
        data_for_each_class.append(np_class_data)
        labels_for_each_class.append(np_class_labels)

    data_as_array = torch.tensor(np.concatenate(data_for_each_class)).transpose(1,3).float()
    labels_as_array = torch.tensor(np.concatenate(labels_for_each_class)).long()

    return CifarLikeDataset(data_as_array,labels_as_array)
