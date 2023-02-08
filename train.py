import torch
from time import time
from scipy.stats import entropy as np_entropy
from dl_utils.label_funcs import label_counts, accuracy, get_trans_dict
from dl_utils.tensor_funcs import cudify
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.distributions import Categorical
import cl_args
from pdb import set_trace
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class ClusterNet(nn.Module):
    def __init__(self,ARGS):
        super().__init__()
        self.bs_train = ARGS.batch_size_train
        self.bs_val = ARGS.batch_size_val
        self.nc = ARGS.nc
        self.nz = ARGS.nz
        self.sigma = ARGS.sigma
        self.prior = ARGS.prior
        self.log_prior = np.log(ARGS.prior)

        if ARGS.imt:
            out_conv_shape = 13
        elif ARGS.stl:
            out_conv_shape = 21
        elif ARGS.fashmnist:
            out_conv_shape = 4
        else:
            out_conv_shape = 5
        nc = 1 if ARGS.fashmnist else 3
        self.conv1 = nn.Conv2d(3, 6, 5)
        if ARGS.arch == 'alex':
            self.net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
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

        self.opt = optim.Adam(self.net.parameters(),lr=ARGS.lr)

        self.centroids = torch.randn(ARGS.nc,ARGS.nz,requires_grad=True,device='cuda')
        self.ng_opt = optim.Adam([{'params':self.centroids}],lr=1e-3)
        self.cluster_dists = None
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
        self.centroids = sample_feature_vecs.clone().detach().requires_grad_(True)

    def forward(self, inp):
        self.bs = inp.shape[0]
        feature_vecs = self.net(inp)
        self.cluster_dists = (feature_vecs[:,None]-self.centroids).norm(dim=2)
        self.assign_batch()

    def assign_batch(self):
        if ARGS.ng:
            self.cluster_loss,self.batch_assignments = neural_gas_loss(.1*self.cluster_dists+(self.cluster_counts+1).log(),self.temp)
        elif ARGS.var:
            self.assign_batch_var()
        elif ARGS.kl:
            self.assign_batch_kl()
        elif ARGS.sinkhorn:
            self.assign_batch_sinkhorn()
        elif ARGS.no_reg:
            min_dists, self.batch_assignments  = self.cluster_dists.min(axis=1)
            self.cluster_loss = min_dists.mean()
        else:
            self.assign_batch_probabilistic()

        self.raw_counts.index_put_(indices=[self.cluster_dists.argmin(axis=1)],values=torch.ones_like(self.batch_assignments),accumulate=True)
        if ARGS.ng or ARGS.sinkhorn or ARGS.kl:
            self.cluster_counts.index_put_(indices=[self.batch_assignments],values=torch.ones_like(self.batch_assignments),accumulate=True)
        if not ARGS.sinkhorn or ARGS.ng:
            self.soft_counts = (-self.cluster_dists).softmax(axis=1).sum(axis=0).detach()
        self.total_soft_counts += self.soft_counts

    def assign_batch_sinkhorn(self):
        with torch.no_grad():
            soft_assignments = sinkhorn(-self.cluster_dists,eps=.5,niters=15)
        if self.prior != 'uniform' and self.epoch_num>0 and ARGS.help_sinkhorn:
            soft_assignments *= torch.tensor(self.prior).cuda().exp()
        self.batch_assignments = soft_assignments.argmin(axis=1)
        self.soft_counts = soft_assignments.sum(axis=0).detach()
        if ARGS.soft_train:
            softmax_probs = (-self.cluster_dists.detach()).softmax(axis=1)
            self.cluster_loss = (self.cluster_dists * softmax_probs).mean()
        else:
            self.cluster_loss = self.cluster_dists[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_var(self):
        self.batch_assignments = self.cluster_dists.argmin(axis=1)
        #self.cluster_loss = 10*(self.cluster_dists**2).sum()
        if ARGS.var_improved:
            self.cluster_loss = 10*self.cluster_dists.mean(axis=0).var()
        else:
            self.cluster_loss = 10*(self.cluster_dists.mean(axis=0)**2).mean()
        self.cluster_loss +=.1*self.cluster_dists[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_kl(self):
        self.batch_assignments = self.cluster_dists.argmin(axis=1)
        self.cluster_loss = 100*-Categorical(self.cluster_dists.mean(axis=0)).entropy()
        if ARGS.kl_cent:
            self.cluster_loss +=.1*self.cluster_dists[torch.arange(self.bs),self.batch_assignments].mean()
        else:
            self.cluster_loss += .1*Categorical(self.cluster_dists).entropy().mean()

    def assign_batch_probabilistic(self):
        flat_x = self.cluster_dists.transpose(0,1).flatten(1).transpose(0,1)
        assigned_key_order = []
        self.batch_assignments = torch.zeros_like(self.cluster_dists[:,0]).long()
        unassigned_idxs = torch.ones_like(flat_x[:,0]).bool()
        cost_table = flat_x/(2*ARGS.sigma)
        if self.prior != 'uniform' and self.epoch_num > 10:
            cost_table -= np.log(self.translated_prior)#*self.acc
        had_repeats = False
        if not self.training and not ARGS.constrained_eval:
            self.batch_assignments = self.cluster_dists.argmin(axis=1)
            return
        assign_iter = 0
        while unassigned_idxs.any():
            try:assert (~unassigned_idxs).sum() == assign_iter or had_repeats
            except: set_trace()
            cost = (cost_table[unassigned_idxs]+(self.cluster_counts+1).log()).min()
            nzs = ((cost_table+(self.cluster_counts+1).log() == cost)*unassigned_idxs[:,None]).nonzero()
            if len(nzs)!=1: had_repeats = True
            new_vec_idx, new_assigned_key = nzs[0]
            if not unassigned_idxs[new_vec_idx]: set_trace()
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
                        #if (rc := self.raw_counts[k].item()) == 0:
                            #continue
                        print(f"{k} constrained: {v.item()}\traw: {self.raw_counts[k].item()}\tsoft: {self.soft_counts[k].item():.3f}")
                if not ARGS.suppress_prints:
                    print(f'batch index: {i}\tloss: {running_loss/10:.3f}')
                running_loss = 0.0
            running_loss += self.cluster_loss.item()
            if (self.centroids==0).all(): set_trace()

    def train_epochs(self,num_epochs,dset,val_too=True):
        if self.prior == 'uniform':
            trainloader = DataLoader(dset,batch_size=self.bs_train,shuffle=True,num_workers=8)
            testloader = DataLoader(dset,batch_size=self.bs_val,shuffle=False,num_workers=8)
        else:
            trainloader = DataLoader(dset,batch_size=self.bs_train,shuffle=True,num_workers=8)
            testloader = DataLoader(dset,batch_size=self.bs_val,shuffle=False,num_workers=8)
        net_backup = self.net
        best_nmi = 0
        if ARGS.warm_start:
            self.init_keys_as_dpoints(trainloader)
        for epoch_num in range(num_epochs):
            self.epoch_num = epoch_num
            self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
            self.train_one_epoch(trainloader)
            if val_too:
                gt = testloader.dataset.targets
                self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
                with torch.no_grad():
                    pred_array = self.test_epoch_unsupervised(testloader)
                num_of_each_label = label_counts(pred_array)
                epoch_hard_counts = np.zeros(self.nc)
                for ass,num in num_of_each_label.items():
                    epoch_hard_counts[ass] = num
                #epoch_hard_counts = np.array(list(num_of_each_label.values()))
                epoch_soft_counts = self.total_soft_counts.detach().cpu().numpy()
                self.trans_dict = get_trans_dict(np.array(gt),pred_array)
                self.acc = (np.array([self.trans_dict[a] for a in gt])==pred_array).mean()
                #assert acc == accuracy(pred_array,np.array(gt))
                idx_array = np.array(list(self.trans_dict.keys())[:-1])
                self.translated_prior = cudify(self.prior[idx_array])
                nmi = normalized_mutual_info_score(pred_array,np.array(gt))
                ari = adjusted_rand_score(pred_array,np.array(gt))
                hce = np_entropy(epoch_hard_counts,base=2)
                sce = np_entropy(epoch_soft_counts,base=2)
                hcv = epoch_hard_counts.var()/epoch_hard_counts.mean()
                scv = epoch_soft_counts.var()/epoch_hard_counts.mean()
                model_distribution = epoch_hard_counts/epoch_hard_counts.sum()
                log_quot = np.log((model_distribution/self.prior)+1e-8)
                kl_star = np.dot(model_distribution,log_quot)
                if max(hcv,scv) > len(dset) - len(dset)/self.nc: set_trace()
                #print({k:v for k,v in num_of_each_label.items() if v < 5})
                #print(f"KL*: {kl_star:.4f}")
                #print(f"Epoch: {epoch_num}\tAcc: {self.acc:.3f}\tNMI: {nmi:.3f}\tARI: {ari:.3f}")
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_acc = self.acc
                    best_ari = ari
                    best_kl_star = kl_star
        print(f"Best Acc: {best_acc:.3f}\tBest NMI: {best_nmi:.3f}\tBest ARI: {best_ari:.3f}\tBest KL*:{best_kl_star}")

    def test_epoch_unsupervised(self,testloader):
        self.eval()
        preds = []
        for i,data in enumerate(testloader):
            images, labels = data
            self(images.cuda())
            preds.append(self.batch_assignments.detach().cpu().numpy())
        pred_array = np.concatenate(preds)
        return pred_array

def neural_gas_loss(v,temp):
    n_instances, n_clusters = v.shape
    weightings = (-torch.arange(n_clusters,device=v.device)/temp).exp()
    sorted_v, assignments_order = torch.sort(v)
    assert (sorted_v**2 * weightings).mean() < ((sorted_v**2).mean() * weightings.mean())
    return (sorted_v**2 * weightings).sum(axis=1), assignments_order[:,0]
def sinkhorn(scores, eps=0.05, niters=3):
    Q = torch.exp(scores / eps).T
    Q /= sum(Q)
    K, B = Q.shape
    r, c = torch.ones(K,device=Q.device) / K, torch.ones(B,device=Q.device) / B
    for _ in range(niters):
        u = torch.sum(Q, dim=1)
        Q *= (r / u).unsqueeze(1)
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    return (Q / torch.sum(Q, dim=0, keepdim=True)).T

if __name__ == '__main__':
    ARGS,dataset = cl_args.get_cl_args_and_dset()

    with torch.autograd.set_detect_anomaly(True):
        cluster_net = ClusterNet(ARGS).cuda()
        cluster_net.train_epochs(ARGS.epochs,dataset,val_too=True)
