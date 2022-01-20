import torch
import gc
from scipy.stats import entropy as np_entropy
from torch.utils.tensorboard import SummaryWriter
from dl_utils.label_funcs import label_counts, accuracy
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import cl_args
from pdb import set_trace
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ClusterNet(nn.Module):
    def __init__(self,nc1,nc2,nc3,writer,temperature):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.opt = optim.Adam(self.parameters())

        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3

        self.k1s = torch.randn(nc1,6,requires_grad=True,device='cuda')
        self.k2s = torch.randn(nc2,16,requires_grad=True,device='cuda')
        self.k3s = torch.randn(nc3,120,requires_grad=True,device='cuda')

        self.ng_opt = optim.Adam([{'params':self.k1s},{'params':self.k2s},{'params':self.k3s}])

        self.k3_counts = torch.zeros(nc3,device='cuda').int()
        self.k3_raw_counts = torch.zeros(nc3,device='cuda').int()
        self.total_soft_counts = torch.zeros(nc3,device='cuda')

        self.act_logits1 = None
        self.act_logits2 = None
        self.act_logits3 = None

        self.writer = writer
        self.epoch_num = -1
        self.temperature = temperature

    def reset_scores(self):
        self.k1_counts = [0]*self.nc1
        self.k2_counts = torch.zeros(self.nc2,device='cuda').int()
        self.k3_counts = torch.zeros(self.nc3,device='cuda').int()
        self.k3_raw_counts = torch.zeros(self.nc3,device='cuda').int()

    def init_keys_as_dpoints(self,dloader):
        #transform = Compose([ToTensor(),Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        #rand_idxs = torch.randint(len(dset.data),size=(self.nc3,))
        inp,targets = next(iter(dloader))
        inp = inp[:self.nc3]
        act1 = self.pool(F.relu(self.bn1(self.conv1(torch.tensor(inp,device='cuda')))))
        act2 = self.pool(F.relu(self.bn2(self.conv2(act1))))
        act3 = self.fc1(torch.flatten(act2, 1))
        self.k3s = act3.clone().detach().requires_grad_(True)

    def forward(self, inp):
        eve_loss = 0
        act1 = self.bn1(self.conv1(inp))
        #self.act_logits1 = (act1[:,None]-self.k1s[:,:,None,None]).norm(dim=2)
        x = self.pool(F.relu(act1))
        act2 = self.bn2(self.conv2(x))
        #self.act_logits2 = (act2[:,None]-self.k2s[:,:,None,None]).norm(dim=2)
        x = self.pool(F.relu(act2))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        act3 = self.fc1(x)
        self.act_logits3 = (act3[:,None]-self.k3s).norm(dim=2)
        x = F.relu(act3)
        act4 = self.fc2(x)
        if ARGS.ng:
            cluster_loss,assignments = self.assign_keys_ng(3)
        elif ARGS.iterative:
            cluster_loss,assignments = self.assign_keys_iterative(3)
        elif ARGS.prob_approx:
            cluster_loss,assignments = self.assign_keys_probabilistic_approx(3)
        elif ARGS.sinkhorn:
            #with torch.no_grad():
                #soft_assignments = sinkhorn(-self.act_logits3,eps=1.)
            soft_assignments = self.act_logits3.softmax(axis=1)
            assignments = soft_assignments.argmin(axis=1)
            cluster_loss = (self.act_logits3*soft_assignments)
            for ass in assignments:
                self.k3_counts[ass] += 1
        elif ARGS.kl:
            cluster_loss = 100*-Categorical(self.act_logits3.mean(axis=0)).entropy()
            cluster_loss += 0.1*Categorical(self.act_logits3).entropy().mean()
            assignments = self.act_logits3.argmin(axis=1)
            for ass in assignments:
                self.k3_counts[ass] += 1
        else:
            cluster_loss,assignments = self.assign_keys_probabilistic(3)
        for a in self.act_logits3.argmin(axis=1):
            self.k3_raw_counts[a]+=1
        self.soft_counts = self.act_logits3.softmax(axis=1).mean(axis=0).detach()
        self.total_soft_counts += self.soft_counts
        return assignments,cluster_loss.mean(),act4

    def assign_keys_ng(self,layer_num):
        if layer_num == 1:
            x,num_keys = self.act_logits1, self.nc1
            counts = self.k1_counts
        if layer_num == 2:
            x,num_keys = self.act_logits2, self.nc2
            counts = self.k2_counts
        if layer_num == 3:
            x,num_keys = self.act_logits3, self.nc3
            counts = self.k3_counts
        #print(self.k3s[:,0])
        cost,assignments = neural_gas_loss(.001*x+(counts+1).log(),self.temperature)
        for a in assignments:
            counts[a] += 1
        return cost, assignments

    def assign_keys_probabilistic_approx(self,layer_num):
        if layer_num == 1:
            x,num_keys = self.act_logits1, self.nc1
            counts = self.k1_counts
        if layer_num == 2:
            x,num_keys = self.act_logits2, self.nc2
            counts = self.k2_counts
        if layer_num == 3:
            x,num_keys = self.act_logits3, self.nc3
            counts = self.k3_counts
        neg_cost_table = 0.1*x.transpose(0,1).flatten(1).transpose(0,1)
        assignments = torch.zeros_like(x[:,0]).long()
        cost,assignments = (neg_cost_table + (counts+1).log()).max(axis=1)
        for ass in assignments:
            counts[ass] += 1
        return -cost, assignments

    def assign_keys_iterative(self,layer_num):
        if layer_num == 1:
            x,num_keys = self.act_logits1, self.nc1
            counts = self.k1_counts
        if layer_num == 2:
            x,num_keys = self.act_logits2, self.nc2
            counts = self.k2_counts
        if layer_num == 3:
            x,num_keys = self.act_logits3, self.nc3
            counts = self.k3_counts
        flat_x = x.transpose(0,1).flatten(1).transpose(0,1)
        assignments = []
        cost_table = 0.1*flat_x
        if not self.training:
            return torch.zeros_like(x[:,0]), x.argmin(axis=1)
        for assign_row in range(len(flat_x)):
            new_assigned_key = (assign_row+(counts+1).log()).argmin()
            assignments.append(new_assigned_key)
            counts[new_assigned_key] += 1
        return cost_table[torch.arange(len(assignments)),assignments],assignments

    def assign_keys_probabilistic(self,layer_num):
        if layer_num == 1:
            x,num_keys = self.act_logits1, self.nc1
            counts = self.k1_counts
        if layer_num == 2:
            x,num_keys = self.act_logits2, self.nc2
            counts = self.k2_counts
        if layer_num == 3:
            x,num_keys = self.act_logits3, self.nc3
            counts = self.k3_counts
        flat_x = x.transpose(0,1).flatten(1).transpose(0,1)
        assigned_key_order = []
        assignments = torch.zeros_like(x[:,0]).long()
        unassigned_idxs = torch.ones_like(flat_x[:,0]).bool()
        cost_table = 0.1*flat_x
        had_repeats = False
        if not self.training:
            return torch.zeros_like(x[:,0]), x.argmin(axis=1)
        for assign_iter in range(len(flat_x)):
            try:assert (~unassigned_idxs).sum() == assign_iter  or had_repeats
            except: set_trace()
            cost = (cost_table[unassigned_idxs]+(counts+1).log()).min()
            nzs = (cost_table+(counts+1).log() == cost).nonzero()
            if len(nzs)!=1: had_repeats = True
            new_vec_idx, new_assigned_key = nzs[0]
            unassigned_idxs[new_vec_idx] = False
            assigned_key_order.append(new_vec_idx)
            assignments[new_vec_idx] = new_assigned_key
            assert cost > 0
            counts[new_assigned_key] += 1
        return cost_table[torch.arange(len(assignments)),assignments],assignments

    def train_one_epoch(self,trainloader,epoch_num):
        self.train()
        running_loss = 0.0
        ng_running_loss = 0.0
        eve_running_loss = 0.0
        for i, data in enumerate(trainloader):
            self.reset_scores()
            inputs, labels = data
            if i==ARGS.db_at: set_trace()
            assignments,ng_loss,act4 = self(inputs.cuda())
            unsupervised_loss = ng_loss
            writer.add_scalar('Loss',unsupervised_loss,i + len(trainloader.dataset)*epoch_num)
            unsupervised_loss.backward()
            self.opt.step()
            self.ng_opt.step()
            self.opt.zero_grad(); self.ng_opt.zero_grad()
            if i % 10 == 0:
                if ARGS.track_counts:
                    for k,v in enumerate(self.k3_counts):
                        print(k,v.item(),self.k3_raw_counts[k].item(),round(self.soft_counts[k].item(),4))
                print(f'batch index: {i}\tloss: {running_loss/10:.3f}')
                running_loss = 0.0
                ng_running_loss = 0.0
                eve_running_loss = 0.0
            running_loss += unsupervised_loss.item()
            ng_running_loss += ng_loss.item()
            if (self.k1s==0).all() or (self.k2s==0).all() or (self.k3s==0).all(): set_trace()

    def train_epochs(self,num_epochs,dset,val_too=True):
        trainloader = DataLoader(dset,batch_size=ARGS.batch_size_train,shuffle=True,num_workers=8)
        testloader = DataLoader(dset, batch_size=ARGS.batch_size_val,shuffle=False,num_workers=8)
        if ARGS.init_from_dpoints:
            self.init_keys_as_dpoints(trainloader)
        for epoch_num in range(num_epochs):
            self.total_soft_counts = torch.zeros_like(self.total_soft_counts)
            self.train_one_epoch(trainloader,epoch_num)
            if val_too:
                gt = testloader.dataset.targets
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
                print(f"Hard counts entropy: {hce:.4f}\tSoft counts entropy: {sce:.4f}")
                print(f"Epoch: {epoch_num}\tAcc: {acc:.3f}\tNMI: {nmi:.3f}\tARI: {ari:.3f}")

    def test_epoch_unsupervised(self,testloader):
        self.eval()
        preds = []
        for i,data in enumerate(testloader):
            images, labels = data
            predicted,ng_loss,act4 = self(images.cuda())
            preds.append(predicted.detach().cpu().numpy())
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
get_dset_fn = torchvision.datasets.CIFAR100 if ARGS.C100 else torchvision.datasets.CIFAR10
transform = Compose([ToTensor(),Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

testset = get_dset_fn(root='./data', train=False, download=True, transform=transform)
if ARGS.test:
    trainset = testset
    trainset.data = trainset.data[:1000]
    trainset.targets = trainset.targets[:1000]
elif ARGS.semitest:
    trainset = testset
    rand_idxs = torch.randint(len(trainset),size=(10000,))
    trainset.data = trainset.data[rand_idxs]
    trainset.targets = torch.tensor(trainset.targets)[rand_idxs].tolist()
else:
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset.data = np.concatenate([trainset.data,testset.data])
trainset.targets = np.concatenate([trainset.targets,testset.targets])

writer = SummaryWriter()
with torch.autograd.set_detect_anomaly(True):
    cluster_net = ClusterNet(ARGS.nc1,ARGS.nc2,ARGS.nc3,writer,temperature=ARGS.temperature).cuda()
    cluster_net.train_epochs(ARGS.epochs,trainset,val_too=True)
