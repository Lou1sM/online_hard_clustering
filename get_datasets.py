import torch
from os.path import join
import torchvision
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from functools import partial
from dl_utils.torch_misc import CifarLikeDataset
from dl_utils.tensor_funcs import numpyify


def get_tweets(test_level):
    X = np.load('datasets/tweets/roberta_doc_vecs.npy')
    y = np.load('datasets/tweets/cluster_labels.npy')
    return CifarLikeDataset(X,y)

def get_cifar10(is_test):
    transform = Compose([ToTensor(),Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    data_dir = '~/dataset/cifar10_data'
    dset = torchvision.datasets.CIFAR10(root=data_dir,train=not is_test,transform=transform,download=True)
    return dset.data, dset.targets

def get_cifar100(is_test):
    transform = Compose([ToTensor(),Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761))])
    data_dir = '~/datasets/cifar100_data'
    dset = torchvision.datasets.CIFAR100(root=data_dir,train=not is_test,transform=transform,download=True)
    return dset.data, dset.targets

def get_fashmnist(is_test):
    transform = Compose([ToTensor()])
    data_dir = '~/dataset/fashmnist_data'
    dset = torchvision.datasets.FashionMNIST(root=data_dir,train=not is_test,transform=transform,download=True)
    return dset.data, dset.targets

def get_stl(test_level):
    transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = './dataset/stl_data'
    def wrapper(dset_func):
        def inner():
            dset = dset_func()
            return CifarLikeDataset(np.transpose(dset.data,(0,3,2,1)),dset.labels,transform)
        return inner
    if test_level == 0:
        dset = torchvision.datasets.STL10(root=data_dir,split='train',transform=transform,download=True)
    else:
        dset = torchvision.datasets.STL10(root=data_dir,split='test',transform=transform,download=True)

    return np.transpose(dset.data,(0,3,2,1)), dset.labels

def get_train_or_test_dset(dset_name,is_test):
    if dset_name=='c10':
        X,y = get_cifar10(is_test)
    elif dset_name=='c100':
        X,y = get_cifar100(is_test)
    elif dset_name=='stl':
        X,y = get_stl(True)
    elif dset_name=='fashmnist':
        X,y = get_fashmnist(is_test)
    X = numpyify(X)
    y = numpyify(y)
    return X, y

def get_dset(dset_name,test_level):
    X, y = get_train_or_test_dset(dset_name,True)
    if test_level==2:
        X = X[:1000]
        y = y[:1000]
    elif test_level==1:
        rand_idxs = np.random.randint(len(X),size=(10000,))
        X = X[rand_idxs]
        y = y[rand_idxs]
    else:
        X_tr, y_tr = get_train_or_test_dset(dset_name,False)
        X = np.concatenate([X_tr,X])
        y = np.concatenate([y_tr,y])
    if dset_name == 'fashmnist':
        transform = ToTensor()
    else:
        transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return CifarLikeDataset(X,y,transform=transform)

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
