import argparse


def get_cl_args():
    parser = argparse.ArgumentParser()
    train_type_group = parser.add_mutually_exclusive_group()
    train_type_group.add_argument('--prob',action='store_true')
    train_type_group.add_argument('--kl',action='store_true')
    train_type_group.add_argument('--parallel',action='store_true')
    train_type_group.add_argument('--ng',action='store_true')
    train_type_group.add_argument('--iterative',action='store_true')
    train_type_group.add_argument('--direct_assign',action='store_true')
    train_type_group.add_argument('--no_cluster_loss',action='store_true')
    train_type_group.add_argument('--sinkhorn',action='store_true')
    dset_group = parser.add_mutually_exclusive_group()
    dset_group.add_argument('--imt',action='store_true')
    dset_group.add_argument('--c100',action='store_true')
    parser.add_argument('--batch_size_train',type=int,default=256)
    parser.add_argument('--batch_size_val',type=int,default=1024)
    parser.add_argument('--warm_start',action='store_true')
    parser.add_argument('--constrained_eval',action='store_true')
    parser.add_argument('--db_at',type=int,default=-1)
    parser.add_argument('--nc',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--temp',type=float,default=1.)
    parser.add_argument('--sigma',type=float,default=5.)
    parser.add_argument('--track_counts',action='store_true')
    parser.add_argument('--test_level','-t',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--arch',type=str,choices=['alex','res'])
    ARGS = parser.parse_args()
    return ARGS

RELEVANT_ARGS = []
