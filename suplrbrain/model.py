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

class SparseAutoencoder(nn.Module):
    """A convolutional k-Sparse autoencoder."""

    @staticmethod
    def get_default_config():
        config = {
            "filters": 64,
            "kernel_size": 4,
            "stride": 2,

            "use_bias": True,
            "use_tied_weights": True,
            "use_lifetime_sparsity": True,

            "encoder_padding": 0,
            "decoder_padding": 0,

            "encoder_nonlinearity": "leaky_relu",
            "decoder_nonlinearity": "sigmoid",

            "sparsity": 3,
            "sparsity_percent": 0.3,
            "sparsity_output_factor": 0.75
        }
        return config

    def __init__(self,
                    num_input_channels: int, 
                    base_channel_size: int, 
                    latent_dim: int,
                    num_classes: int,
                    act_fn: object = nn.GELU, 
                    k=None, 
                    per_k=None):
        super(SparseAutoencoder, self).__init__()
        self.config = self.get_default_config()
        if k is not None:
            self.config['sparsity'] = k
        if per_k is not None:
            self.config['sparsity_percent'] = per_k
        self.enc_out_dim = 512
        self.input_height = 32
        self.latent_dim = latent_dim
        self.encoder = ResNet9Enc(False, False)
        self.decoder = ResNet9Dec(self.latent_dim, self.input_height,
                                        False, False)
        self.latent = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, num_classes),
            nn.ReLU(),
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
        encoding = self.encoder(x)
        encoding = self.latent(encoding)
        output = self.fc(encoding)
        decoding = self.decoder(encoding)
        return output, decoding

if __name__ == '__main__':
    inp = torch.randn((2, 1, 32, 32), requires_grad=True)
    m = SparseAutoencoder()
    # print(inp)
    out = m(inp)

    print(out[0].shape)
    print(out[1].shape)