from time import time
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from os.path import join
from dl_utils.label_funcs import label_counts, get_trans_dict, accuracy
from dl_utils.tensor_funcs import cudify, numpyify
from dl_utils.misc import set_experiment_dir, asMinutes, scatter_clusters
from dl_utils.torch_misc import CifarLikeDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import cl_args
from pdb import set_trace
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from HAR.nets import EncByLayer


class ClusterNet(nn.Module):
    def __init__(self,ARGS):
        super().__init__()
        self.bs_train = ARGS.batch_size_train
        self.bs_val = ARGS.batch_size_val
        self.nc = ARGS.nc
        self.nz = ARGS.nz
        #self.sigma = ARGS.sigma
        if ARGS.dataset == 'tweets':
            counts = np.load('datasets/tweets/cluster_label_counts.npy')
            self.log_prior = np.log(counts/counts.sum())
        else:
            self.prior = ARGS.prior
        self.log_prior = np.log(self.prior)

        if ARGS.dataset == 'imt':
            out_conv_shape = 13
        elif ARGS.dataset == 'stl':
            out_conv_shape = 21
        elif ARGS.dataset == 'fashmnist':
            out_conv_shape = 4
        else:
            out_conv_shape = 5
        nc = 1 if ARGS.dataset == 'fashmnist' else 3
        self.conv1 = nn.Conv2d(3, 6, 5)
        if ARGS.arch == 'alex':
            self.net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
            self.net.classifier = self.net.classifier[:5] # remove final linear and relu
            self.net.classifier[4] = nn.Linear(4096,self.nz,device='cuda')
        if ARGS.arch == 'res':
            self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            self.net.fc.weight.data = self.net.fc.weight.data[:self.nz]
            self.net.fc.bias.data = self.net.fc.bias.data[:self.nz]
            self.net.fc.out_features = self.nz
        elif ARGS.arch == 'simp':
            self.net = nn.Sequential(
                nn.Conv2d(nc, 6, 5),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(1),
                nn.Linear(16 * out_conv_shape * out_conv_shape, self.nz),
                )
        elif ARGS.arch == 'fc':
            self.net = nn.Sequential(
                nn.Linear(768,ARGS.hidden_dim),
                nn.ReLU(),
                nn.Linear(ARGS.hidden_dim,self.nz))
        elif ARGS.arch == '1dcnn':
            x_filters = (50,40,7,4)
            x_strides = (2,2,1,1)
            max_pools = ((2,1),(2,1),(2,1),(2,1))
            y_filters = (1,1,1,117)
            y_strides = (1,1,1,1)
            self.net = nn.Sequential(EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,False).cuda(),nn.Flatten())

        self.opt = optim.Adam(self.net.parameters(),lr=ARGS.lr)

        self.centroids = torch.randn(ARGS.nc,ARGS.nz,requires_grad=True,device='cuda',dtype=torch.double)
        self.inv_covars = (1/ARGS.sigma)*torch.eye(ARGS.nz,requires_grad=ARGS.is_train_covars,device='cuda',dtype=torch.double).repeat(self.nc,1,1)
        self.half_log_det_inv_covars = torch.log(torch.tensor(ARGS.sigma))*self.nz/2
        #self.inv_covars = torch.randn(ARGS.nc,ARGS.nz,ARGS.nz,requires_grad=True,device='cuda',dtype=torch.double)
        self.ng_opt = optim.Adam([{'params':self.centroids}],lr=ARGS.lr)
        if ARGS.is_train_covars:
            self.ng_opt.add_param_group({'params':self.inv_covars})
        self.cluster_log_probs = None
        self.cluster_counts = torch.zeros(ARGS.nc,device='cuda').long()
        self.raw_counts = torch.zeros(ARGS.nc,device='cuda').long()
        self.total_soft_counts = torch.zeros(ARGS.nc,device='cuda')

        self.epoch_num = -1
        self.temp = ARGS.temp

        self.training = True

    def train(self):
        self.training = True
        self.net.train()
        self.centroids.requires_grad = True

    def eval(self):
        self.training = False
        self.net.eval()
        self.centroids.requires_grad = False

    def reset_scores(self):
        self.cluster_counts = torch.zeros(self.nc,device='cuda').long()
        self.raw_counts = torch.zeros(self.nc,device='cuda').long()

    def init_keys_as_dpoints(self,dloader):
        self.eval()
        inp,targets = next(iter(dloader))
        inp = inp[:self.nc]
        while len(inp) < self.nc:
            new_inp,targets = next(iter(dloader))
            inp = torch.cat([inp,new_inp[:self.nc-len(inp)]])
        sample_feature_vecs = self.net(inp.cuda())
        self.centroids = sample_feature_vecs.clone().detach().double().requires_grad_(True)

    def forward(self, inp):
        self.bs = inp.shape[0]
        feature_vecs = self.net(inp)
        cluster_dists = feature_vecs[:,None]-self.centroids
        self.cluster_log_probs = torch.einsum('bcu,czu,bcz->bc',cluster_dists,self.inv_covars,cluster_dists) - self.half_log_det_inv_covars
        #assert torch.allclose(self.cluster_log_probs, cluster_dists.norm(dim=2)**2)
        #assert all([torch.allclose(self.cluster_log_probs[b,c],(cluster_dists[b,c]@self.inv_covars[c]) @cluster_dists[b,c]) for b in range(self.bs) for c in range(self.nc)])
        self.assign_batch()
        return feature_vecs

    def assign_batch(self):
        if ARGS.ng:
            self.cluster_loss,self.batch_assignments = neural_gas_loss(.1*self.cluster_log_probs+(self.cluster_counts+1).log(),self.temp)
        elif ARGS.var:
            self.assign_batch_var()
        elif ARGS.kl:
            self.assign_batch_kl()
        elif ARGS.sinkhorn:
            self.assign_batch_sinkhorn()
        elif ARGS.no_reg:
            min_dists, self.batch_assignments = self.cluster_log_probs.min(axis=1)
            self.cluster_loss = min_dists.mean()
        else:
            self.assign_batch_probabilistic()

        self.raw_counts.index_put_(indices=[self.cluster_log_probs.argmin(axis=1)],values=torch.ones_like(self.batch_assignments),accumulate=True)
        if ARGS.ng or ARGS.sinkhorn or ARGS.kl:
            self.cluster_counts.index_put_(indices=[self.batch_assignments],values=torch.ones_like(self.batch_assignments),accumulate=True)
        if not ARGS.sinkhorn or ARGS.ng:
            self.soft_counts = (-self.cluster_log_probs).softmax(axis=1).sum(axis=0).detach()
        self.total_soft_counts += self.soft_counts

    def assign_batch_sinkhorn(self):
        with torch.no_grad():
            #hard_counts = (torch.arange(self.nc).cuda() == self.cluster_log_probs.argmax(axis=1,keepdims=True)).float()
            soft_assignments = sinkhorn(-self.cluster_log_probs,is_hard_reg=ARGS.hard_sinkhorn,eps=.5,niters=15)
        if self.prior != 'uniform' and self.epoch_num>0 and ARGS.help_sinkhorn:
            soft_assignments *= torch.tensor(self.translated_prior).cuda().exp()
        self.batch_assignments = soft_assignments.argmin(axis=1)
        self.soft_counts = soft_assignments.sum(axis=0).detach()
        if ARGS.soft_train:
            softmax_probs = (-self.cluster_log_probs.detach()).softmax(axis=1)
            self.cluster_loss = (self.cluster_log_probs * softmax_probs).mean()
        else:
            self.cluster_loss = self.cluster_log_probs[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_var(self):
        self.batch_assignments = self.cluster_log_probs.argmin(axis=1)
        #self.cluster_loss = 10*(self.cluster_log_probs**2).sum()
        if ARGS.var_improved:
            self.cluster_loss = 10*self.cluster_log_probs.mean(axis=0).var()
        else:
            self.cluster_loss = 10*(self.cluster_log_probs.mean(axis=0)**2).mean()
        self.cluster_loss +=.1*self.cluster_log_probs[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_kl(self):
        self.batch_assignments = self.cluster_log_probs.argmin(axis=1)
        self.cluster_loss = 100*-Categorical(self.cluster_log_probs.mean(axis=0)).entropy()
        if ARGS.kl_cent:
            self.cluster_loss +=.1*self.cluster_log_probs[torch.arange(self.bs),self.batch_assignments].mean()
        else:
            self.cluster_loss += .1*Categorical(self.cluster_log_probs).entropy().mean()

    def assign_batch_probabilistic(self):
        assigned_key_order = []
        cost_table = self.cluster_log_probs.transpose(0,1).flatten(1).transpose(0,1)
        self.batch_assignments = torch.zeros_like(self.cluster_log_probs[:,0]).long()
        unassigned_idxs = torch.ones_like(cost_table[:,0]).bool()
        #cost_table = flat_x/(2*ARGS.sigma)
        if ARGS.imbalance > 0 and self.epoch_num > 0:
            cost_table -= self.translated_log_prior
        had_repeats = False
        if not self.training and not ARGS.constrained_eval:
            self.batch_assignments = self.cluster_log_probs.argmin(axis=1)
            return
        assign_iter = 0
        while unassigned_idxs.any():
            assert (~unassigned_idxs).sum() == assign_iter or had_repeats
            cost = (cost_table[unassigned_idxs]+(self.cluster_counts+1).log()).min()
            nzs = ((cost_table+(self.cluster_counts+1).log() == cost)*unassigned_idxs[:,None]).nonzero()
            if len(nzs)!=1: had_repeats = True
            new_vec_idx, new_assigned_key = nzs[0]
            assert unassigned_idxs[new_vec_idx]
            unassigned_idxs[new_vec_idx] = False
            assigned_key_order.append(new_vec_idx)
            self.batch_assignments[new_vec_idx] = new_assigned_key
            self.cluster_counts[new_assigned_key] += 1
            #assert cost > 0
            assign_iter += 1
        self.cluster_loss = cost_table[torch.arange(self.bs),self.batch_assignments].mean()

    def train_one_epoch(self,trainloader):
        self.train()
        running_loss = 0.0
        self.reset_scores()
        for i, data in enumerate(trainloader):
            if not ARGS.keep_scores:
                self.reset_scores()
            self.batch_inputs, self.batch_labels = data
            if i==ARGS.db_at: set_trace()
            self(self.batch_inputs.cuda())
            self.cluster_loss.backward()
            self.opt.step()
            self.ng_opt.step()
            self.opt.zero_grad(); self.ng_opt.zero_grad()
            if i % 10 == 0 and i > 0:
                if ARGS.track_counts:
                    for k,v in enumerate(self.cluster_counts):
                        if (rc := self.raw_counts[k].item()) == 0:
                            continue
                        print(f"{k} constrained: {v.item()}\traw: {self.raw_counts[k].item()}\tsoft: {self.soft_counts[k].item():.3f}")
                if not ARGS.suppress_prints:
                    print(f'batch index: {i}\tloss: {running_loss/10:.3f}')
                running_loss = 0.0
            running_loss += self.cluster_loss.item()
            if (self.centroids==0).all(): set_trace()
            if ARGS.is_test > 0:
                break

    def train_epochs(self,num_epochs,dset,val_too=True):
        trainloader = DataLoader(dset,batch_size=self.bs_train,shuffle=True,num_workers=8)
        testloader = DataLoader(dset,batch_size=self.bs_val,shuffle=False,num_workers=8)
        best_acc = -1
        best_nmi = -1
        best_ari = -1
        best_kl_star = -1
        best_linear_probe_acc = -1
        best_knn_probe_acc = -1
        if ARGS.warm_start:
            self.init_keys_as_dpoints(trainloader)
        for epoch_num in range(num_epochs):
            self.epoch_num = epoch_num
            self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
            self.train_one_epoch(trainloader)
            if val_too:
                self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
                with torch.no_grad():
                    self.test_epoch_unsupervised(testloader)
                model_distribution = self.epoch_hard_counts/self.epoch_hard_counts.sum()
                log_quot = np.log((model_distribution/self.prior)+1e-8)
                self.kl_star = np.dot(model_distribution,log_quot)
                if self.nmi > best_nmi:
                    best_nmi = self.nmi
                    best_acc = self.acc
                    best_ari = self.ari
                    best_kl_star = self.kl_star
                else:
                    with torch.no_grad():
                        self.test_epoch_unsupervised(testloader)
                linear_probe_acc, knn_probe_acc = self.train_test_probes(dset)
                if linear_probe_acc > best_linear_probe_acc:
                    best_linear_probe_acc = linear_probe_acc
                    best_knn_probe_acc = knn_probe_acc
        print(f"Best Acc: {best_acc:.3f}\tBest NMI: {best_nmi:.3f}\tBest ARI: {best_ari:.3f}\tBest KL*:{best_kl_star:.5f}\tBest linear probe acc:{best_linear_probe_acc:.3f}\tBest KNN probe acc:{best_knn_probe_acc:.3f}")
        with open(join(ARGS.exp_dir,'ARGS.txt'),'w') as f:
            f.write(f'Dataset: {ARGS.dataset}\n')
            for a in ['batch_size_train','nz','hidden_dim','lr','sigma']:
                f.write(f'{a}: {getattr(ARGS,a)}\n')
            f.write(f'warm_start: {ARGS.warm_start}\n')

        with open(join(ARGS.exp_dir,'results.txt'),'w') as f:
            f.write(f'ACC: {best_acc:.3f}\nNMI: {best_nmi:.3f}\nARI: {best_ari:.3f}\n')
            f.write(f'KL-star: {best_kl_star:.3f}\nLin-Acc: {best_linear_probe_acc:.3f}\nKNN-Acc: {best_knn_probe_acc:.3f}\n')

    def train_test_probes(self,dset):
        self.eval()
        dloader = DataLoader(dset,batch_size=self.bs_val,shuffle=False,num_workers=8)
        all_encodings = []
        for i,data in enumerate(dloader):
            images, labels = data
            encodings = numpyify(self.net(images.cuda()))
            all_encodings.append(encodings)
        X = np.concatenate(all_encodings)
        y = dset.targets
        X_tr,X_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.33)
        lin_reg = LogisticRegression().fit(X_tr,y_tr)
        lin_test_preds = lin_reg.predict(X_ts)
        lin_test_acc = (lin_test_preds==y_ts).mean()
        knn_reg = KNeighborsClassifier(n_neighbors=ARGS.n_neighbors).fit(X_tr,y_tr)
        knn_test_preds = knn_reg.predict(X_ts)
        knn_test_acc = (knn_test_preds==y_ts).mean()
        return lin_test_acc, knn_test_acc

    def test_epoch_unsupervised(self,testloader):
        self.eval()
        preds = []
        all_feature_vecs = []
        data_for_clusters = [[] for _ in range(self.nc)]
        for images,labels in testloader:
            feature_vecs = self(images.cuda())
            all_feature_vecs.append(numpyify(feature_vecs))
            assignments =self.batch_assignments
            for cluster_idx in range(self.nc):
                data_for_clusters[cluster_idx].append(feature_vecs[assignments==cluster_idx])
            preds.append(assignments.detach().cpu().numpy())
        pred_array = np.concatenate(preds)
        num_of_each_label = label_counts(pred_array)
        self.epoch_hard_counts = np.zeros(self.nc)
        for ass,num in num_of_each_label.items():
            self.epoch_hard_counts[ass] = num
        if ARGS.estimate_covars and len(num_of_each_label) == self.nc: # don't set covars if some dpoints missing
            unnormed_inv_covars = torch.stack([torch.inverse(torch.cat(cd).T.cov()) for cd in data_for_clusters])
            self.inv_covars = (unnormed_inv_covars*self.inv_covars.mean()/unnormed_inv_covars.mean()).double()
            if unnormed_inv_covars.isnan().any():
                breakpoint()
        self.epoch_soft_counts = self.total_soft_counts.detach().cpu().numpy()
        self.gt = testloader.dataset.targets
        self.trans_dict = get_trans_dict(np.array(self.gt),pred_array)
        self.acc = accuracy(pred_array,np.array(self.gt))
        if self.acc == 0:
            breakpoint()
        idx_array = np.array(list(self.trans_dict.keys())[:-1])
        self.translated_prior = self.prior[idx_array]
        self.translated_log_prior = cudify(self.log_prior[idx_array])
        self.nmi = normalized_mutual_info_score(pred_array,np.array(self.gt))
        self.ari = adjusted_rand_score(pred_array,np.array(self.gt))
        self.hcv = self.epoch_hard_counts.var()/self.epoch_hard_counts.mean()
        self.scv = self.epoch_soft_counts.var()/self.epoch_hard_counts.mean()
        if ARGS.viz_clusters:
            feature_vecs_array = np.concatenate(all_feature_vecs)
            import umap
            to_viz = umap.UMAP().fit_transform(feature_vecs_array)
            ax = scatter_clusters(to_viz,testloader.dataset.targets)
            breakpoint()

def neural_gas_loss(v,temp):
    n_instances, n_clusters = v.shape
    weightings = (-torch.arange(n_clusters,device=v.device)/temp).exp()
    sorted_v, assignments_order = torch.sort(v)
    assert (sorted_v**2 * weightings).mean() < ((sorted_v**2).mean() * weightings.mean())
    return (sorted_v**2 * weightings).sum(axis=1), assignments_order[:,0]

def sinkhorn(scores, is_hard_reg=False, eps=0.05, niters=3):
    Q = torch.exp(scores / eps).T
    #Q = torch.softmax(scores / eps,1).T
    Q /= sum(Q)
    eps2 = 0.1
    if is_hard_reg:
        hard_counts = (torch.arange(Q.shape[0]).cuda()[:,None] == Q.argmax(axis=0)).float()
        #Q = torch.cat([Q,1*hard_counts.sum(axis=1,keepdims=True)],axis=1)
        Q = (Q + eps2*hard_counts) / (1+eps2)
    K, B = Q.shape
    r, c = torch.ones(K,device=Q.device) / K, torch.ones(B,device=Q.device) / B
    for _ in range(niters):
        u = torch.sum(Q, dim=1)
        Q *= (r / u).unsqueeze(1)
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    if is_hard_reg:
        Q = (Q*2) - hard_counts
        #Q = Q[:,:-1]
    return (Q / torch.sum(Q, dim=0, keepdim=True)).T

if __name__ == '__main__':
    ARGS,dataset = cl_args.get_cl_args_and_dset()
    ARGS.exp_dir = set_experiment_dir(f'experiments/{ARGS.expname}',name_of_trials='experiments/tmp',overwrite=ARGS.overwrite)
    start_time = time()
    cluster_net = ClusterNet(ARGS).cuda()
    cluster_net.train_epochs(ARGS.epochs,dataset,val_too=True)
    print(f'Total time: {asMinutes(time()-start_time)}')
