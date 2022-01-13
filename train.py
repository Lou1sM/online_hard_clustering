import torch
from dl_utils.label_funcs import label_counts
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cl_args
from pdb import set_trace


ARGS = cl_args.get_cl_args()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

if ARGS.test:
    trainset.data = trainset.data[:1000]
    trainset.targets = trainset.targets[:1000]
trainloader = torch.utils.data.DataLoader(trainset,batch_size=ARGS.batch_size,shuffle=True,num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=ARGS.batch_size,shuffle=False,num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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
    def __init__(self,nc1,nc2,nc3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.opt = optim.Adam(self.parameters())

        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3

        self.k1s_raw = torch.randn(nc1,6,requires_grad=True,device='cuda')
        self.k2s_raw = torch.randn(nc2,16,requires_grad=True,device='cuda')
        self.k3s_raw = torch.randn(nc3,120,requires_grad=True,device='cuda')

        self.v1s = torch.randn(nc1,6,requires_grad=True,device='cuda')
        self.v2s = torch.randn(nc2,16,requires_grad=True,device='cuda')
        self.v3s = torch.randn(nc3,120,requires_grad=True,device='cuda')

        self.ng_opt = optim.Adam([{'params':self.k1s_raw},{'params':self.k2s_raw},{'params':self.k3s_raw}])
        self.opt.add_param_group({'params':self.v1s})
        self.opt.add_param_group({'params':self.v2s})
        self.opt.add_param_group({'params':self.v3s})

        self.k1_counts = [0]*nc1

        self.act_logits1 = None
        self.act_logits2 = None
        self.act_logits3 = None

    def reset_scores(self):
        self.k1_counts = [0]*self.nc1

    def forward(self, inp):
        #processed1 = self.pool(self.bn1(self.conv1(inp)))
        processed1 = self.pool(self.conv1(inp))
        self.act_logits1 = -(processed1[:,None]-self.k1s_raw[:,:,None,None]).norm(dim=2)
        if ARGS.track_counts:
            for k,v in label_counts(self.act_logits1.argmax(dim=1).flatten().numpy()).items():
                self.k1_counts[k] += v
        #self.act_logits1 = torch.einsum('bijk,li->bljk',processed1,k1s_normed)
        weighted_val1 = torch.einsum('bljk,li->bijk',(self.act_logits1).softmax(dim=1),self.v1s)

        processed2 = self.pool(F.relu(self.conv2(weighted_val1)))
        #self.act_logits2 = torch.einsum('bijk,li->bljk',processed2,k2s_normed)
        self.act_logits2 = -(processed2[:,None]-self.k2s_raw[:,:,None,None]).norm(dim=2)
        weighted_val2 = torch.einsum('bljk,li->bijk',(self.act_logits2).softmax(dim=1),self.v2s)

        processed3 = self.fc1(torch.flatten(weighted_val2, 1)) # flatten all dimensions except batch
        #self.act_logits3 = processed3 @ k3s_normed.transpose(0,1)
        self.act_logits3 = -(processed3[:,None]-self.k3s_raw).norm(dim=2)
        weighted_val3 = self.act_logits3.softmax(dim=1) @ self.v3s

        outp = self.fc2(weighted_val3)
        l1,l2 = self.ng_losses()
        if self.training:
            (l1*0.1 + 0.01*l2).backward(retain_graph=True);
            self.opt.zero_grad() # Think this is right..
        return outp,l1,l2

    def ng_losses(self):
        l1 = 0
        l2 = 0
        for s in (self.act_logits1,self.act_logits2,self.act_logits3):
            scores_by_key = s.transpose(0,1).flatten(1)
            l1 += us_cross_entropy_logits(scores_by_key)
            l2 += open_eve(scores_by_key)
        return l1, l2

    def train_epoch(self,trainloader):
        running_loss = 0.0
        ng_running_loss = 0.0
        eve_running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data

            outputs,ng_loss,eve_loss = self(inputs.cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            loss.backward(retain_graph=True)
            (ng_loss + 0.1*eve_loss).backward()

            self.opt.step()
            self.opt.zero_grad()
            self.ng_opt.step(); self.ng_opt.zero_grad()
            if (i+1) % 50 == 0:
                #if ARGS.track_counts:
                    #for k,v in enumerate(self.k1_counts):
                        #if v>0: print(k,v)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} ng_loss: {ng_running_loss / 2000:.3f} eve_loss: {eve_running_loss / 2000:.3f}')
                running_loss = 0.0
                ng_running_loss = 0.0
                eve_running_loss = 0.0
                self.reset_scores()
            running_loss += loss.item()
            ng_running_loss += ng_loss.item()
            eve_running_loss += eve_loss.item()
            if (self.k1s_raw==0).all() or (self.k2s_raw==0).all() or (self.k3s_raw==0).all(): set_trace()

    def test_epoch(testloader):
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs,ng_loss,eve_loss = net(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
        return correct/total

def us_cross_entropy_logits(scores):
    if (scores.max(axis=1)[0] > 0).any(): set_trace()
    #print(scores.max(axis=1)[0].mean())
    #return -(scores.max(axis=1)[0] - scores.logsumexp(axis=1)).mean()
    return -scores.max(axis=1)[0].mean()

def open_eve(scores):
    eve_diff = (scores.mean(axis=1).var(unbiased=False) - scores.var(axis=1,unbiased=False).mean())
    eve_diff_as_frac = eve_diff / scores.var(unbiased=False)
    if not eve_diff_as_frac <= 1: set_trace()
    return eve_diff_as_frac


cluster_net = ClusterNet(196,30,10).cuda()
net = Net().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images.cuda())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print(total)
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
