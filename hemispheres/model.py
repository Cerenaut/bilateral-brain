import torch
import torch.nn as nn

import math
import utils

from pl_bolts.models.autoencoders.components \
    import resnet18_decoder, resnet18_encoder

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class SparseConv(nn.Module):
    def __init__(self):
        super(SparseConv, self).__init__()
    
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

    def filter(self, x, k, k_percent):  # Applied to a batch.
        """ """
        x = x.permute(1, 0, 2, 3)
        inp = x.reshape(x.shape[0] * x.shape[1], -1)
        inp = self.batch_topk(inp, k)
        inp = inp.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        k_factor = math.ceil(k_percent * k * x.shape[1])
        inp = self.batch_topk(inp, k_factor)
        inp = inp.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        inp = inp.permute(1, 0, 2, 3)
        return inp

    def forward(self, x, k, k_percent):
        x = self.filter(x, k, k_percent)
        return x

def conv_block(in_channels, out_channels, act_fn = nn.GELU, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              act_fn()]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, act_fn=nn.GELU):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64, act_fn=act_fn)
        self.conv2 = conv_block(64, 128, act_fn=act_fn, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, act_fn=act_fn),
                                    conv_block(128, 128, act_fn=act_fn))
        
        self.conv3 = conv_block(128, 256, act_fn=act_fn, pool=True)
        self.conv4 = conv_block(256, 512, act_fn=act_fn, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512, act_fn=act_fn),
                                    conv_block(512, 512, act_fn=act_fn))
        self.fc = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.fc(out)
        # print(out.shape)
        return out

class SparseAutoencoder(nn.Module):
    """A convolutional k-Sparse autoencoder."""

    def __init__(self,
                    in_channels: int,
                    num_classes: int,
                    act_fn: object = nn.ReLU, 
                    k=None, 
                    per_k=None):
        super(SparseAutoencoder, self).__init__()
        if k is not None:
            self.config['sparsity'] = k
        if per_k is not None:
            self.config['sparsity_percent'] = per_k
        self.enc_out_dim = 512
        self.encoder = ResNet9(in_channels=in_channels, act_fn=act_fn)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.enc_out_dim, num_classes),
            act_fn(),
        )

    def reset_parameters(self):
        # self.apply(lambda m: utils.initialize_parameters(m, weight_init='xavier_normal_', bias_init='zeros_'))

        # Similar initialization to TF implementation of ConvAEs
        def custom_weight_init(m):
            if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                return

            if m.weight is not None:
                utils.truncated_normal_(m.weight, mean=0.0, std=0.03)

            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        self.apply(custom_weight_init)

    def forward(self, x):  # pylint: disable=arguments-differ
        output = self.encoder(x)
        return output

class Combiner(nn.Module):
    def __init__(self,
                    enc_out_dim: int = 1024,
                    act_fn: object = nn.ReLU):
        super(Combiner, self).__init__()

        self.fc = nn.Sequential(
                                nn.Dropout(0.3),
                                nn.Linear(enc_out_dim, 256),
                                act_fn())
        self.broad = nn.Sequential(
                                nn.Dropout(0.3),
                                nn.Linear(256, 20),
                                act_fn())
        self.narrow = nn.Sequential(
                                nn.Dropout(0.3),
                                nn.Linear(256, 100),
                                act_fn())

    def reset_parameters(self):
        def custom_weight_init(m):
            if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                return

            if m.weight is not None:
                utils.truncated_normal_(m.weight, mean=0.0, std=0.03)

            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        self.apply(custom_weight_init)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.fc(x)
        broad = self.broad(x)
        narrow = self.narrow(x)
        return narrow, broad