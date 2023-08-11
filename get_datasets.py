import torch
from os.path import join
import torchvision
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from functools import partial
from dl_utils.torch_misc import CifarLikeDataset
from dl_utils.tensor_funcs import numpyify
from HAR.make_dsets import make_realdisp_dset
from HAR.project_config import realdisp_info


def get_tweets(is_use_testset):
    X = np.load('datasets/tweets/roberta_doc_vecs.npy')
    y = np.load('datasets/tweets/cluster_labels.npy')
    return CifarLikeDataset(X,y)

def get_cifar10(is_use_testset):
    transform = Compose([ToTensor(),Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    data_dir = '~/dataset/cifar10_data'
    dset = torchvision.datasets.CIFAR10(root=data_dir,train=not is_use_testset,transform=transform,download=True)
    return dset.data, dset.targets

def get_cifar100(is_use_testset):
    transform = Compose([ToTensor(),Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761))])
    data_dir = '~/datasets/cifar100_data'
    dset = torchvision.datasets.CIFAR100(root=data_dir,train=not is_use_testset,transform=transform,download=True)
    return dset.data, dset.targets

def get_fashmnist(is_use_testset):
    transform = Compose([ToTensor()])
    data_dir = '~/dataset/fashmnist_data'
    dset = torchvision.datasets.FashionMNIST(root=data_dir,train=not is_use_testset,transform=transform,download=True)
    return dset.data, dset.targets

def get_stl(is_test_run):
    transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = './dataset/stl_data'
    def wrapper(dset_func):
        def inner():
            dset = dset_func()
            return CifarLikeDataset(np.transpose(dset.data,(0,3,2,1)),dset.labels,transform)
        return inner
    if is_test_run:
        dset = torchvision.datasets.STL10(root=data_dir,split='train',transform=transform,download=True)
    else:
        dset = torchvision.datasets.STL10(root=data_dir,split='test',transform=transform,download=True)

    return np.transpose(dset.data,(0,3,2,1)), dset.labels

def get_train_or_test_dset(dset_name,is_use_testset):
    if dset_name=='c10':
        X,y = get_cifar10(is_use_testset)
    elif dset_name=='c100':
        X,y = get_cifar100(is_use_testset)
    elif dset_name=='stl':
        X,y = get_stl(True)
    elif dset_name=='fashmnist':
        X,y = get_fashmnist(is_use_testset)
    elif dset_name=='imt':
        X,y = get_imagenet_tiny(is_use_testset)
    else:
        print(f'\nUNRECOGNIZED DATASET: {dset_name}\n')
    X = numpyify(X)
    y = numpyify(y)
    return X, y

def get_dset(dset_name,is_test_run):
    if dset_name=='realdisp':
        subj_ids = realdisp_info().possible_subj_ids
        if is_test_run:
            subj_ids = subj_ids[:1]
        dset,_ = make_realdisp_dset(step_size=5,window_size=512,subj_ids=subj_ids)
        #big_realdisp_data = np.load('datasets/big_realdisp_X.npy')
        #big_realdisp_targets = np.load('datasets/big_realdisp_y.npy')
        #dset,_ = CifarLikeDataset(big_realdisp_data,big_realdisp_targets)
        return dset
    else:
        X, y = get_train_or_test_dset(dset_name,True)
    if is_test_run:
        X = X[:1000]
        y = y[:1000]
    elif dset_name!='imt': # no train-test split in im-tiny
        X_tr, y_tr = get_train_or_test_dset(dset_name,False)
        X = np.concatenate([X_tr,X])
        y = np.concatenate([y_tr,y])
    if dset_name == 'fashmnist':
        transform = ToTensor()
    elif dset_name == 'imt':
        transform = None
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

    return data_as_array,labels_as_array
