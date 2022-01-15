import argparse
import sys


def get_cl_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test','-t',action='store_true')
    group.add_argument('--semitest','-st',action='store_true')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--eve',action='store_true')
    parser.add_argument('--nc1',type=int,default=15)
    parser.add_argument('--nc2',type=int,default=12)
    parser.add_argument('--nc3',type=int,default=10)
    parser.add_argument('--track_counts',action='store_true')
    ARGS = parser.parse_args()
    return ARGS

RELEVANT_ARGS = ['ablate_label_filter','clusterer','dset','no_umap','num_meta_epochs','num_meta_meta_epochs','num_pseudo_label_epochs','reinit','step_size','subject_independent']
