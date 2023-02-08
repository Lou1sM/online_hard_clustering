import argparse
from dl_utils.torch_misc import CifarLikeDataset
import numpy as np
import get_datasets


RELEVANT_ARGS = []
def get_cl_args():
    parser = argparse.ArgumentParser()
    train_type_group = parser.add_mutually_exclusive_group()
    train_type_group.add_argument('--prob',action='store_true')
    train_type_group.add_argument('--kl',action='store_true')
    train_type_group.add_argument('--var',action='store_true')
    train_type_group.add_argument('--ng',action='store_true')
    train_type_group.add_argument('--no_reg',action='store_true')
    train_type_group.add_argument('--no_cluster_loss',action='store_true')
    train_type_group.add_argument('--sinkhorn',action='store_true')
    dset_group = parser.add_mutually_exclusive_group()
    dset_group.add_argument('--imt',action='store_true')
    dset_group.add_argument('--c100',action='store_true')
    dset_group.add_argument('--svhn',action='store_true')
    dset_group.add_argument('--stl',action='store_true')
    dset_group.add_argument('--fashmnist',action='store_true')
    dset_group.add_argument('--tweets',action='store_true')
    parser.add_argument('--batch_size_train',type=int,default=256)
    parser.add_argument('--batch_size_val',type=int,default=1024)
    parser.add_argument('--warm_start',action='store_true')
    parser.add_argument('--constrained_eval',action='store_true')
    parser.add_argument('--db_at',type=int,default=-1)
    parser.add_argument('--nc',type=int,default=10)
    parser.add_argument('--nz',type=int,default=128)
    parser.add_argument('--hidden_dim',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--temp',type=float,default=1.)
    parser.add_argument('--sigma',type=float,default=5.)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--track_counts',action='store_true')
    parser.add_argument('--keep_scores',action='store_true')
    parser.add_argument('--soft_train',action='store_true')
    parser.add_argument('--kl_cent',action='store_true')
    parser.add_argument('--var_improved',action='store_true')
    parser.add_argument('--suppress_prints',action='store_true')
    parser.add_argument('--help_sinkhorn',action='store_true')
    parser.add_argument('--test_level','-t',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--imbalance',type=int,default=0)
    parser.add_argument('--arch',type=str,choices=['alex','res','simp','fc'],default='simp')
    ARGS = parser.parse_args()
    return ARGS

def make_dset_imbalanced(dset,class_probs,nc):
    imbalanced_data = []
    imbalanced_targets = []
    for i,p in enumerate(class_probs):
        targets = np.array(dset.targets)
        label_mask = targets==i
        rand_mask =np.random.rand(sum(label_mask))<p
        new_data = dset.data[label_mask][rand_mask]
        new_targets = targets[label_mask][rand_mask]
        imbalanced_data.append(new_data)
        imbalanced_targets.append(new_targets)
    imbalanced_data_arr = np.concatenate(imbalanced_data)
    imbalanced_targets_arr = np.concatenate(imbalanced_targets)
    assert len(imbalanced_data_arr) == len(imbalanced_targets_arr)
    return CifarLikeDataset(imbalanced_data_arr,imbalanced_targets_arr,transform=dset.transform)

def get_cl_args_and_dset():
    args = get_cl_args()
    if args.imbalance==1:
        class_probs=np.array([1.0,1.0,1.0,1.0,1.0,1.0,0.95,0.9,0.85,0.8])
    elif args.imbalance==2:
        class_probs=np.array([1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55])
    elif args.imbalance==3:
        class_probs=np.array([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])

    if args.imt:
        print('using dset imt')
        dataset = get_datasets.get_imagenet_tiny(args.test_level)
        args.nc = 200
    elif args.fashmnist:
        print('using dset fashmnist')
        dataset = get_datasets.get_fashmnist(args.test_level)
        args.nc = 10
    elif args.stl:
        print('using dset stl')
        dataset = get_datasets.get_stl(args.test_level)
        args.nc = 10
    elif args.c100:
        print('using dset c100')
        dataset = get_datasets.get_cifar100(args.test_level)
        args.nc = 100
    elif args.tweets:
        print('using dset tweets')
        dataset = get_datasets.get_tweets(args.test_level)
        args.nc = 269
        args.arch = 'fc'
    else:
        print('using dset c10')
        dataset = get_datasets.get_cifar10(args.test_level)
        args.nc = 10

    if args.imbalance > 0:
        dataset = make_dset_imbalanced(dataset,class_probs,args.nc)
        args.prior = class_probs/class_probs.sum()
    else:
        args.prior = np.ones(args.nc)/args.nc
    return args, dataset
