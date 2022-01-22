import argparse
import sys


def get_cl_args():
    parser = argparse.ArgumentParser()
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test','-t',action='store_true')
    test_group.add_argument('--semitest','-st',action='store_true')
    train_type_group = parser.add_mutually_exclusive_group()
    train_type_group.add_argument('--prob',action='store_true')
    train_type_group.add_argument('--prob_approx',action='store_true')
    train_type_group.add_argument('--kl',action='store_true')
    train_type_group.add_argument('--ng',action='store_true')
    train_type_group.add_argument('--entropy',action='store_true')
    train_type_group.add_argument('--iterative',action='store_true')
    train_type_group.add_argument('--direct_assign',action='store_true')
    train_type_group.add_argument('--no_cluster_loss',action='store_true')
    train_type_group.add_argument('--sinkhorn',action='store_true')
    dset_group = parser.add_mutually_exclusive_group()
    dset_group.add_argument('--ImageNet',action='store_true')
    dset_group.add_argument('--C100',action='store_true')
    parser.add_argument('--batch_size_train',type=int,default=256)
    parser.add_argument('--batch_size_val',type=int,default=1024)
    parser.add_argument('--eve',action='store_true')
    parser.add_argument('--warm_start',action='store_true')
    parser.add_argument('--db_at',type=int,default=-1)
    parser.add_argument('--nc1',type=int,default=15)
    parser.add_argument('--nc2',type=int,default=12)
    parser.add_argument('--nc3',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--temperature',type=float,default=1.)
    parser.add_argument('--track_counts',action='store_true')
    ARGS = parser.parse_args()
    return ARGS

RELEVANT_ARGS = []
