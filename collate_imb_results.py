import os
import numpy as np
import pandas as pd

dsets = ('Cifar10','Cifar100','FMNIST','STL','RealDisp')
metrics = ('acc','nmi','ari','KL')
methods = ('ours','sk','ent','ss','ckm')
df = pd.DataFrame(columns=('imb1','imb2','imb3'), index=pd.MultiIndex.from_product(dsets,metrics))
for method_num,method in enumerate(methods):
    for imb in range(1,4):
        for dset_num,dset in enumerate(dsets):
            accs = np.empty(5)
            nmis = np.empty(5)
            aris = np.empty(5)
            klss = np.empty(5)
            for iter_num in range(5):
                with open(f'experiments/{dset_num+1}.{method_num}.{iter_num}.imb{imb}') as f:
                    d = f.readlines()
                for line in d:
                    if line.startswith('ACC'):
                        accs[iter_num] = float(line[5:])
                    elif line.startswith('NMI'):
                        nmis[iter_num] = float(line[5:])
                    elif line.startswith('ARI'):
                        nmis[iter_num] = float(line[5:])
                    elif line.startswith('Hard KL-star'):
                        klss[iter_num] = float(line[5:])
            df.loc[f'imb{imb}',dset_num,'acc'] = accs.mean()
            df.loc[f'imb{imb}',dset_num,'nmi'] = nmis.mean()
            df.loc[f'imb{imb}',dset_num,'ari'] = aris.mean()
            df.loc[f'imb{imb}',dset_num,'kls'] = klss.mean()
    df.to_latex(f'{method}_imb_results.txt')
