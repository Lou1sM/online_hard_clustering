import torch
from os.path import join
import torchvision
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from functools import partial
from dl_utils.torch_misc import CifarLikeDataset
from dl_utils.tensor_funcs import numpyify


def get_cifar10(test_level):
    transform = Compose([ToTensor(),Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    data_dir = '~/dataset/cifar10_data'
    torch_cifar10_func_train = partial(torchvision.datasets.CIFAR10,root=data_dir,train=True,transform=transform,download=True)
    torch_cifar10_func_test = partial(torchvision.datasets.CIFAR10,root=data_dir,train=False,transform=transform,download=True)
    return get_torch_available_dset(torch_cifar10_func_train,torch_cifar10_func_test,test_level)

def get_tweets(test_level):
    X = np.load('datasets/tweets/roberta_doc_vecs.npy')
    y = np.load('datasets/tweets/cluster_labels.npy')
    return CifarLikeDataset(X,y)

def get_cifar100(test_level):
    transform = Compose([ToTensor(),Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761))])
    data_dir = '~/datasets/cifar100_data'
    torch_cifar100_func_train = partial(torchvision.datasets.CIFAR100,root=data_dir,train=True,transform=transform,download=True)
    torch_cifar100_func_test = partial(torchvision.datasets.CIFAR100,root=data_dir,train=False,transform=transform,download=True)
    return get_torch_available_dset(torch_cifar100_func_train,torch_cifar100_func_test,test_level)

def get_fashmnist(test_level):
    add_colour_dim = lambda t: t.unsqueeze(0)
    transform = Compose([ToTensor()])
    data_dir = '~/dataset/fashmnist_data'
    dset = torchvision.datasets.FashionMNIST(root=data_dir,train=True,transform=transform)
    dl = data.DataLoader(dset,128)
    return dset

def get_svhn(test_level):
    transform = Compose([ToTensor(),Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
    data_dir = '~/dataset/svhn_data'
    def wrapper(dset_func):
        def inner():
            dset = dset_func()
            return CifarLikeDataset(np.transpose(dset.data,(0,3,2,1)),dset.labels,transform)
        return inner
    torch_svhn_func_train_ = partial(torchvision.datasets.SVHN,root=data_dir,split='train',transform=transform,download=True)
    torch_svhn_func_test_ = partial(torchvision.datasets.SVHN,root=data_dir,split='test',transform=transform,download=True)
    torch_svhn_func_train = wrapper(torch_svhn_func_train_)
    torch_svhn_func_test = wrapper(torch_svhn_func_test_)
    return get_torch_available_dset(torch_svhn_func_train,torch_svhn_func_test,test_level)

def get_stl(test_level):
    transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = './dataset/stl_data'
    def wrapper(dset_func):
        def inner():
            dset = dset_func()
            return CifarLikeDataset(np.transpose(dset.data,(0,3,2,1)),dset.labels,transform)
        return inner
    torch_stl_func_train_ = partial(torchvision.datasets.STL10,root=data_dir,split='train',transform=transform,download=True)
    torch_stl_func_test_ = partial(torchvision.datasets.STL10,root=data_dir,split='test',transform=transform,download=True)
    torch_stl_func_train = wrapper(torch_stl_func_train_)
    torch_stl_func_test = wrapper(torch_stl_func_test_)
    return get_torch_available_dset(torch_stl_func_train,torch_stl_func_test,test_level)

def get_torch_available_dset(torch_dset_func_train,torch_dset_func_test,test_level):
    testset = torch_dset_func_test()
    if test_level==2:
        trainset = testset
        trainset.data = trainset.data[:1000]
        trainset.targets = trainset.targets[:1000]
    elif test_level==1:
        trainset = testset
        rand_idxs = torch.randint(len(trainset),size=(10000,))
        trainset.data = trainset.data[rand_idxs]
        trainset.targets = torch.tensor(trainset.targets)[rand_idxs].tolist()
    else:
        trainset = torch_dset_func_train()
        trainset.data = np.concatenate([trainset.data,testset.data])
        trainset.targets = trainset.targets + testset.targets
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
