import torch
import datasets
from scipy.stats import entropy as np_entropy
from torch.utils.tensorboard import SummaryWriter
from dl_utils.label_funcs import label_counts, accuracy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import cl_args
from pdb import set_trace
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class ClusterNet(nn.Module):
    def __init__(self,nc,bs_train,bs_val,writer,temp,arch):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        if arch == 'alex':
            self.net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        if arch == 'res':
            self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.opt = optim.Adam(self.net.parameters())

        self.bs_train = bs_train
        self.bs_val = bs_val
        self.nc = nc

        self.centroids = torch.randn(nc,1000,requires_grad=True,device='cuda')
        self.ng_opt = optim.Adam([{'params':self.centroids}],lr=1.)
        self.cluster_dists = None
        self.cluster_counts = torch.zeros(nc,device='cuda').int()
        self.raw_counts = torch.zeros(nc,device='cuda').int()
        self.total_soft_counts = torch.zeros(nc,device='cuda')

        self.writer = writer
        self.epoch_num = -1
        self.temp = temp

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
        self.cluster_counts = torch.zeros(self.nc,device='cuda').int()
        self.raw_counts = torch.zeros(self.nc,device='cuda').int()

    def init_keys_as_dpoints(self,dloader):
        self.eval()
        inp,targets = next(iter(dloader))
        inp = inp[:self.nc]
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
        elif ARGS.iterative:
            self.assign_batch_iterative()
        elif ARGS.parallel:
            self.assign_batch_parallel()
        elif ARGS.kl:
            self.assign_batch_kl()
        elif ARGS.sinkhorn:
            self.assign_batch_sinkhorn()
        else:
            self.assign_batch_probabilistic()

        if ARGS.ng or ARGS.sinkhorn or ARGS.kl:
            for ass in self.batch_assignments:
                self.cluster_counts[ass] += 1
        for ass in self.cluster_dists.argmin(axis=1):
            self.raw_counts[ass]+=1
        if not ARGS.sinkhorn or ARGS.ng:
            self.soft_counts = (-self.cluster_dists).softmax(axis=1).sum(axis=0).detach()
        self.total_soft_counts += self.soft_counts

    def assign_batch_sinkhorn(self):
        with torch.no_grad():
            soft_assignments = sinkhorn(-self.cluster_dists,eps=.5,niters=15)
        self.batch_assignments = soft_assignments.argmin(axis=1)
        self.soft_counts = soft_assignments.sum(axis=0).detach()
        self.cluster_loss = self.cluster_dists[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_kl(self):
        self.cluster_loss = 10*-Categorical(self.cluster_dists.mean(axis=0)).entropy()
        self.cluster_loss += Categorical(self.cluster_dists).entropy().mean()
        self.batch_assignments = self.cluster_dists.argmin(axis=1)

    def assign_batch_parallel(self):
        if not self.training:
            self.batch_assignments = self.cluster_dists.argmin(axis=1)
            return
        neg_cost_table = 0.1*self.cluster_dists.transpose(0,1).flatten(1).transpose(0,1)
        self.batch_assignments = torch.arange(self.nc).tile(self.bs//self.nc)
        leftovers = torch.arange(self.bs % self.nc)
        self.batch_assignments = torch.cat([self.batch_assignments,leftovers]).cuda()
        tentative_counts = torch.tensor([self.bs/self.nc]).tile(self.nc).cuda()
        num_passes = 50
        c = self.nc
        for _ in range(num_passes):
            for chunk_start_idx in range(0,self.bs,c):
                chunk = neg_cost_table[chunk_start_idx:chunk_start_idx+c]
                chunk_assignments = (chunk+(tentative_counts+1).log()).argmin(axis=1)
                for ass in chunk_assignments:
                    tentative_counts[ass] += 1
                for ass in self.batch_assignments[chunk_start_idx:chunk_start_idx+c]:
                    tentative_counts[ass] -= 1
                    assert tentative_counts[ass] >= -1
                assert (tentative_counts.sum() - self.bs).abs() < 1e-3
                self.batch_assignments[chunk_start_idx:chunk_start_idx+c] = chunk_assignments
        self.cluster_loss = self.cluster_dists[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_iterative(self):
        flat_x = self.cluster_dists.transpose(0,1).flatten(1).transpose(0,1)
        self.batch_assignments = []
        cost_table = 0.1*flat_x
        if not self.training:
            return torch.zeros_like(self.cluster_dists[:,0]), self.cluster_dists.argmin(axis=1)
        for assign_row in range(len(flat_x)):
            new_assigned_key = (assign_row+(self.cluster_counts+1).log()).argmin()
            self.batch_assignments.append(new_assigned_key)
            self.cluster_counts[new_assigned_key] += 1
        self.cluster_loss = cost_table[torch.arange(self.bs),self.batch_assignments].mean()

    def assign_batch_probabilistic(self):
        flat_x = self.cluster_dists.transpose(0,1).flatten(1).transpose(0,1)
        assigned_key_order = []
        self.batch_assignments = torch.zeros_like(self.cluster_dists[:,0]).long()
        unassigned_idxs = torch.ones_like(flat_x[:,0]).bool()
        cost_table = flat_x/(2*ARGS.sigma)
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
            assert cost > 0
            assign_iter += 1
        self.cluster_loss = cost_table[torch.arange(self.bs),self.batch_assignments].mean()

    def train_one_epoch(self,trainloader,epoch_num):
        self.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            self.reset_scores()
            inputs, labels = data
            if i==ARGS.db_at: set_trace()
            self(inputs.cuda())
            writer.add_scalar('Loss',self.cluster_loss,i + len(trainloader.dataset)*epoch_num)
            self.cluster_loss.backward()
            self.opt.step()
            self.ng_opt.step()
            self.opt.zero_grad(); self.ng_opt.zero_grad()
            if i % 10 == 0:
                if ARGS.track_counts:
                    for k,v in enumerate(self.cluster_counts):
                        if (rc := self.raw_counts[k].item()) == 0:
                            continue
                        print(f"{k} constrained: {v.item()}\traw: {rc}\tsoft: {self.soft_counts[k].item():.3f}")
                print(f'batch index: {i}\tloss: {running_loss/10:.3f}')
                running_loss = 0.0
            running_loss += self.cluster_loss.item()
            if (self.centroids==0).all(): set_trace()

    def train_epochs(self,num_epochs,dset,val_too=True):
        trainloader = DataLoader(dset,batch_size=self.bs_train,shuffle=True,num_workers=8)
        testloader = DataLoader(dset,batch_size=self.bs_val,shuffle=False,num_workers=8)
        if ARGS.warm_start:
            self.init_keys_as_dpoints(trainloader)
        for epoch_num in range(num_epochs):
            self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
            self.train_one_epoch(trainloader,epoch_num)
            if val_too:
                gt = testloader.dataset.targets
                self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
                with torch.no_grad():
                    pred_array = self.test_epoch_unsupervised(testloader)
                num_of_each_label = label_counts(pred_array)
                epoch_hard_counts = np.array(list(num_of_each_label.values()))
                epoch_soft_counts = self.total_soft_counts.detach().cpu().numpy()
                acc = accuracy(pred_array,np.array(gt))
                nmi = normalized_mutual_info_score(pred_array,np.array(gt))
                ari = adjusted_rand_score(pred_array,np.array(gt))
                hce = np_entropy(epoch_hard_counts)
                sce = np_entropy(epoch_soft_counts)
                hcv = epoch_hard_counts.var()/epoch_hard_counts.mean()
                scv = epoch_soft_counts.var()/epoch_hard_counts.mean()
                print(num_of_each_label)
                print(f"Hard counts entropy: {hce:.4f}\tSoft counts entropy: {sce:.4f}")
                print(f"Hard counts variance: {hcv:.4f}\tSoft counts variance: {scv:.4f}")
                print(f"Epoch: {epoch_num}\tAcc: {acc:.3f}\tNMI: {nmi:.3f}\tARI: {ari:.3f}")

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

ARGS = cl_args.get_cl_args()
if ARGS.imt:
    dataset = datasets.get_imagenet_tiny(ARGS.test_level)
    nc = 200
elif ARGS.c100:
    dataset = datasets.get_cifar100(ARGS.test_level)
    nc = 100
else:
    dataset = datasets.get_cifar10(ARGS.test_level)
    nc = 10

writer = SummaryWriter()
with torch.autograd.set_detect_anomaly(True):
    cluster_net = ClusterNet(nc,writer=writer,bs_train=ARGS.batch_size_train,bs_val=ARGS.batch_size_val,temp=ARGS.temp,arch=ARGS.arch).cuda()
    cluster_net.train_epochs(ARGS.epochs,dataset,val_too=True)
