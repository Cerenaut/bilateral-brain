import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from argparse import Namespace

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
def check_list():
    return ['narrow', 'broad', 'both', 'feature']

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
                    mode='feature',
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
        # modules  = [nn.AvgPool2d(4),
        #             nn.Flatten(),]
        modules  = [nn.Flatten(),]
        self.mode = mode
        if self.mode not in check_list():
            raise Exception('Mode of training does not match')
        if self.mode == 'narrow':
            modules.append(nn.Linear(in_features=512, out_features=100, bias=True))
            # modules.append(sparse_mlp) # sparsity
            self.fc = nn.Sequential(*modules)
        elif self.mode == 'broad':
            modules.append(nn.Linear(in_features=512, out_features=20, bias=True))
            # modules.append(sparse_mlp)
            self.fc = nn.Sequential(*modules)
        elif self.mode == 'both':
            cmodules = modules.copy()
            modules.append(nn.Linear(in_features=512, out_features=100, bias=True))
            modules.append(sparse_mlp)
            self.ffc = nn.Sequential(*modules)
            cmodules.append(nn.Linear(in_features=512, out_features=20, bias=True))
            cmodules.append(sparse_mlp)
            self.cfc = nn.Sequential(*cmodules)
        elif self.mode == 'feature':
            self.fc = nn.Sequential(*modules)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.avg_pool(out)
        if self.mode == 'narrow' or self.mode == 'broad':
            return self.fc(out)
        elif self.mode == 'both':
            return self.ffc(out), self.cfc(out)
        elif self.mode =='feature':
            out = self.fc(out)
            return out


class InvertedResNet9(nn.Module):
    def __init__(self, 
                    mode='feature',
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
        modules  = [nn.AvgPool2d(4), 
                    nn.Flatten(),]
        self.mode = mode
        if self.mode not in check_list():
            raise Exception('Mode of training does not match')
        if self.mode == 'narrow':
            modules.append(sparse_mlp)
            modules.append(nn.Linear(in_features=64, out_features=100, bias=True))
            self.fc = nn.Sequential(*modules)
        elif self.mode == 'broad':
            modules.append(nn.Linear(in_features=64, out_features=20, bias=True))
            modules.append(sparse_mlp)
            self.fc = nn.Sequential(*modules)
        elif self.mode == 'both':
            cmodules = modules.copy()
            modules.append(nn.Linear(in_features=64, out_features=100, bias=True))
            modules.append(sparse_mlp)
            self.ffc = nn.Sequential(*modules)
            cmodules.append(nn.Linear(in_features=64, out_features=100, bias=True))
            cmodules.append(sparse_mlp)
            self.cfc = nn.Sequential(*cmodules)
        elif self.mode == 'feature':
            self.fc = nn.Sequential(*modules)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        if self.mode == 'narrow' or self.mode == 'broad':
            return self.fc(out)
        elif self.mode == 'both':
            return self.ffc(out), self.cfc(out)
        elif self.mode =='feature':
            out = self.fc(out)
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

class BicamNet(nn.Module):
    """
    A Bicameral Residual network.
    """
    def __init__(self,
                    carch: str,
                    farch: str, 
                    mode: str = 'feature',
                    cmodel_path = None,
                    fmodel_path = None,
                    cfreeze_params: bool = False,
                    ffreeze_params: bool = False,
                    bicam_mode: str = 'both'):
        super(BicamNet, self).__init__()
        self.broad = globals()[carch](
                                Namespace(**
                                    {
                                        "mode": mode,
                                        "k": 0.9,
                                        "k_percent": 0.9,
                                    }))
        self.narrow = globals()[farch](
                                Namespace(**
                                    {
                                        "mode": mode,
                                        "k": 0.9,
                                        "k_percent": 0.9,
                                    }))
        if cmodel_path is not None:
            self.broad.load_state_dict(load_model(cmodel_path))
            if cfreeze_params:
                freeze_params(self.broad)
        if fmodel_path is not None:
            self.narrow.load_state_dict(load_model(fmodel_path))
            if ffreeze_params:
                freeze_params(self.narrow)
        out_dim = self.broad.res2[-1][0].out_channels + self.narrow.res2[-1][0].out_channels
        self.fcombiner = nn.Sequential(
                            nn.Dropout(0.6),
                            nn.Linear(out_dim, 100)
                            )
        self.ccombiner = nn.Sequential(
                            nn.Dropout(0.6),
                            nn.Linear(out_dim, 20)
                            )
        self.bicam_mode = bicam_mode

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]
        """
        fembed = self.narrow(x)
        cembed = self.broad(x)
        embed = torch.cat([fembed, cembed], axis=1)
        if self.bicam_mode == 'both':
            foutput = self.fcombiner(embed)
            coutput = self.ccombiner(embed)
            return foutput, coutput
        elif self.bicam_mode == 'feature':
            return embed

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

def load_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict)
    return model

def load_feat_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items() if 'fc' not in k}
    model.load_state_dict(model_dict)
    return model

def freeze_params(model):
    """[summary]

    Args:
        model ([type]): [description]
    """
    for param in model.parameters():
        param.requires_grad = False
    
def bicameral(args):
    """[summary]
    """
    return BicamNet(args.carch,
                    args.farch, 
                    args.mode, 
                    args.cmodel_path, args.fmodel_path, 
                    args.cfreeze_params, args.ffreeze_params,
                    args.bicam_mode)

def load_bicam_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model_', '').replace('encoder.', ''):v for k,v in sdict.items() if not 'combiner' in k and not 'fc' in k}
    fc_dict = {k.replace('combiner.', '').replace('broad.', 'ccombiner.').replace('narrow.', 'fcombiner.'):v for k,v in sdict.items() if 'combiner' in k}
    model_dict.update(fc_dict)
    model.load_state_dict(model_dict)
    return model