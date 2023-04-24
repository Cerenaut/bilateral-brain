import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
class SparseMLP(nn.Module):
    def __init__(self, k, k_percent):
        super(SparseMLP, self).__init__()
        self.k = k
        self.k_percent = k_percent
    
    def unravel_index(self, index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = torch.div(index, dim, rounding_mode='trunc')
        return tuple(reversed(out))

    def forward(self, x):
        k_per_layer = math.ceil(self.k * x.shape[1])
        k_batch = math.ceil(self.k_percent * x.shape[0] * k_per_layer)
        
        _, ind = torch.topk(x, k_per_layer, dim=1)
        a = torch.arange(0, ind.shape[0])
        a = torch.cat(k_per_layer * [a.unsqueeze(-1)], axis=-1)
        ind = [a.view(-1), ind.view(-1)]
        mask = torch.zeros_like(x).bool()
        mask[ind] = 1
        x = x * mask
        x[x == 0] = -float('inf')

        _, ind = torch.topk(x.view(-1), k_batch)
        ind = self.unravel_index(ind, x.shape)
        mask = torch.zeros_like(x).bool()
        mask[ind] = 1
        x = torch.nan_to_num(x, neginf=0)
        x = x * mask
        return x

class SparseConv(nn.Module):
    def __init__(self, k, k_percent):
        super(SparseConv, self).__init__()
        self.k = k
        self.k_percent = k_percent

    def unravel_index(self, mask, indices, k):
        """ """
        
        a = torch.arange(0, indices.shape[0])
        a = torch.cat(k * [a.unsqueeze(-1)], axis=-1)
        indices = [a.view(-1), indices.view(-1)]
        mask[indices] = 1
        return mask

    def batch_topk(self, inp, k):
        """ """ 
        
        (buffer, indices) = torch.topk(inp, k, -1, True)
        mask = torch.zeros_like(inp).bool()
        mask = self.unravel_index(mask, indices, k)
        inp = inp * mask
        return inp

    def filter(self, x):  # Applied to a batch.
        """ """
        k_per_map = math.ceil(self.k * x.shape[2] * x.shape[3]) 
        x = x.permute(1, 0, 2, 3)
        inp = x.reshape(x.shape[0] * x.shape[1], -1)
        inp = self.batch_topk(inp, k_per_map)
        inp = inp.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        k_factor = math.ceil(self.k_percent * k_per_map * x.shape[1])
        inp = self.batch_topk(inp, k_factor)
        inp = inp.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        inp = inp.permute(1, 0, 2, 3)
        return inp

    def forward(self, x):
        x = self.filter(x)
        return x

def conv_block(in_channels, 
                out_channels, 
                k=None, 
                k_percent=None, 
                act_fn = nn.GELU, 
                pool=False):
    if k is not None and k_percent is not None:
        sparse_layer = SparseConv(k, k_percent)
    else:
        sparse_layer=nn.Identity()
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              sparse_layer, # sparsity models
              nn.BatchNorm2d(out_channels),
              act_fn(),]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def conv_trans_block(in_channels, 
                        out_channels, 
                        act_fn = nn.GELU, 
                        pool=False):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1), 
              act_fn(),
              nn.BatchNorm2d(out_channels),]
    if pool: layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self,
                    k=None, 
                    k_percent=None, 
                    act_fn=nn.ReLU):
        super().__init__()
        self.k = k
        self.k_percent = k_percent
        k = None
        k_percent = None
        self.conv1 = conv_block(3, 64, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn)
        self.conv2 = conv_block(64, 128, k=self.k,         
                                k_percent=self.k_percent, act_fn=act_fn, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn),
                                    conv_block(128, 128, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn))
        
        self.conv3 = conv_block(128, 256, k=self.k,         
                                k_percent=self.k_percent, act_fn=act_fn, pool=True)
        self.conv4 = conv_block(256, 512, k=self.k,         
                                k_percent=self.k_percent, act_fn=act_fn, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn),
                                    conv_block(512, 512, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn))
        if k is not None and k_percent is not None:
            sparse_mlp = SparseMLP(k, k_percent)
        else:
            sparse_mlp = nn.Identity()
        self.avg_pool = nn.MaxPool2d(4)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.avg_pool(out)
        return out

class InvertedResNet9(nn.Module):
    def __init__(self, 
                    k=None, 
                    k_percent=None,    
                    act_fn=nn.ReLU,):
        super().__init__()
        self.k = k
        self.k_percent = k_percent
        k = None
        k_percent = None
        self.conv1 = conv_block(3, 512, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn)
        self.conv2 = conv_block(512, 256, k=self.k,         
                                k_percent=self.k_percent, act_fn=act_fn, pool=True)
        self.res1 = nn.Sequential(conv_block(256, 256, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn),
                                    conv_block(256, 256, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn))
        
        self.conv3 = conv_block(256, 128, k=self.k,         
                                k_percent=self.k_percent, act_fn=act_fn, pool=True)
        self.conv4 = conv_block(128, 64, k=self.k,         
                                k_percent=self.k_percent, act_fn=act_fn, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 64, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn),
                                    conv_block(64, 64, k=self.k,         
                                        k_percent=self.k_percent, act_fn=act_fn))
        if k is not None and k_percent is not None:
            sparse_mlp = SparseMLP(k, k_percent)
        else:
            sparse_mlp = nn.Identity()
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return out

class ResNet9Wrapper(nn.Module):
    """
    A Residual network.
    """
    def __init__(self,
                    arch: str, 
                    mode: str = 'feature',
                    model_path = None,
                    freeze_params: bool = False,):
        super(ResNet9, self).__init__()
        self.model = globals()[arch](mode=mode)
        if model_path is not None:
            self.model.load_state_dict(load_model(model_path))
        if freeze_params:
            freeze_params(self.model)

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]
        """
        out = self.model(x)
        return out

def net(args):
    """[summary]
    """
    return ResNet9Wrapper(args.arch, args.mode)

def resnet9(args):
    """ return a ResNet 9 object
    """
    return ResNet9(args.mode, args.k, args.k_percent)

def invresnet9(args):
    """ return a ResNet 18 object
    """
    return InvertedResNet9(args.mode, args.k, args.k_percent)

