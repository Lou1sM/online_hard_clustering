import torch.nn as nn
import math
import torch

class EncByLayer(nn.Module):
    def __init__(self,ncvs,ksizes,strides,paddings,max_pools,show_shapes):
        super(EncByLayer,self).__init__()
        self.show_shapes = show_shapes
        num_layers = len(ksizes)
        assert all([isinstance(x,int) for l in (ncvs,ksizes,strides) for x in l])
        assert all(len(x)==num_layers for x in (ksizes,strides,max_pools))
        conv_layers = []
        for i in range(num_layers):
            if i<num_layers-1:
                conv_layer = nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],ksizes[i],strides[i],paddings[i]),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            else: #No batch norm on the last layer
                conv_layer = nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],ksizes[i],strides[i],paddings[i]),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self,x):
        if self.show_shapes: print(x.shape)
        for i,conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            # sometimes errors in the batchnorm if size is already down to 1
            if self.show_shapes: print(i,x.shape)
        return x

class DecByLayer(nn.Module):
    def __init__(self,ncvs,ksizes,strides,paddings,show_shapes):
        super(DecByLayer,self).__init__()
        self.show_shapes = show_shapes
        n_layers = len(ksizes)
        assert all([isinstance(x,int) for l in (ncvs,ksizes,strides,paddings) for x in l])
        assert all(len(x)==n_layers for x in (ksizes,strides,paddings) )
        conv_trans_layers = [nn.Sequential(
                nn.ConvTranspose2d(ncvs[i],ncvs[i+1],ksizes[i],strides[i],paddings[i]),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                )
            for i in range(n_layers)]
        self.conv_trans_layers = nn.ModuleList(conv_trans_layers)

    def forward(self,x):
        if self.show_shapes: print(x.shape)
        for i,conv_trans_layer in enumerate(self.conv_trans_layers):
            x = conv_trans_layer(x)
            if self.show_shapes: print(i,x.shape)
        return x

def increment_approx_exponentially(insize,outsize,n_increments):
    base = (outsize/insize)**(1/(n_increments-1))
    assert base > 1
    increments = [int(insize*base**(i)) for i in range(n_increments)]
    if increments[-1] != outsize:
        print(f'readjusting output size from {increments[-1]} to {outsize}')
        increments[-1] = outsize
    if increments[0] != insize:
        breakpoint()
    return increments

def build_convt_net(in_chans,in_shape,outsize,n_layers):
    chans = list(reversed(increment_approx_exponentially(in_chans,outsize,n_layers+1)))
    sizes = increment_approx_exponentially(1,in_shape,n_layers+1)
    ksizes,strides,paddings = zip(*[infer_single_layer_shape(sizes[i+1],sizes[i]) for i in range(len(sizes)-1)])
    return DecByLayer(chans,ksizes,strides,paddings,False)

def build_conv_net(in_chans,in_shape,outsize,n_layers):
    chans = increment_approx_exponentially(in_chans,outsize,n_layers+1)
    sizes = increment_approx_exponentially(1,in_shape,n_layers+1) # no flatten
    ksizes,strides,paddings = zip(*[infer_single_layer_shape(sizes[i+1],sizes[i]) for i in reversed(range(len(sizes)-1))])
    max_pools = [1]*len(strides) # no max pool, just strides
    return EncByLayer(chans,ksizes,strides,paddings,max_pools,False)

def infer_single_layer_shape(in_size,out_size):
    stride_size = int(in_size/out_size)
    ksize = in_size - stride_size*(out_size - 1)
    tentative_out_size = (in_size - ksize)/stride_size + 1
    assert tentative_out_size == out_size
    padding = int((out_size - tentative_out_size)/2)
    if ksize<3:
        padding = int(math.ceil((3-ksize)/2))
        ksize += 2*padding
    tentative_out_size = (in_size + 2*padding - ksize)/stride_size + 1
    if not tentative_out_size == out_size:
        breakpoint()
    return ksize,stride_size,padding

if __name__ == '__main__':
    enc = build_conv_net(3,64,256,7)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    print(enc(torch.ones(1,3,64,64)).shape)
    dec = build_convt_net(3,64,256,2)
    print(dec(torch.ones(1,256,1,1)).shape)
