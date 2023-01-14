"""utils.py"""

import os
import math
import random
import datetime
import os.path as osp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

PARENT_PATH = Path(__file__).parent.resolve()
PROJECT_PATH = PARENT_PATH.parent.parent.absolute().resolve()

def run_cli():
    validate_path('./config.yaml')
    with open('./config.yaml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def yaml_func(config_param):

    if isinstance(config_param, list):
        call_list = []
        local_func = locals().keys()
        for param in config_param:
            if param in local_func:
                call_list.append(locals()[param])
        return call_list
    elif isinstance(config_param, dict):
        call = None
        global_func = globals().keys()
        key = config_param['type']
        del config_param['type']
        if key in global_func:
            call = globals()[key](**config_param)
        return call

def inverse_normalize(img,
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std, dtype=img.dtype, device=img.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    img.mul_(std).add_(mean)
    return img


def matplotlib_imshow(img, one_channel=False, unnormalize=False):
    if one_channel:
        img = img.mean(dim=0)
    if unnormalize:
        img = inverse_normalize(img)  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu())
    fig = plt.figure(figsize=(8, 8))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return fig

def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu())
            max_grads.append(p.grad.abs().max().detach().cpu())
    fig = plt.figure(figsize=(8, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=20) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return fig


def plot_classes_preds(decode1, decode2, image1, image2):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        init = idx*4
        ax = fig.add_subplot(4, 4, init+1, xticks=[], yticks=[])
        matplotlib_imshow(decode1[idx], one_channel=False, unnormalize=False)
        ax.set_title("decode1")
        ax = fig.add_subplot(4, 4, init+2, xticks=[], yticks=[])
        matplotlib_imshow(decode2[idx], one_channel=False, unnormalize=False)
        ax.set_title("decode2")
        ax = fig.add_subplot(4, 4, init+3, xticks=[], yticks=[])
        matplotlib_imshow(image1[idx], one_channel=False, unnormalize=False)
        ax.set_title("img1")
        ax = fig.add_subplot(4, 4, init+4, xticks=[], yticks=[])
        matplotlib_imshow(image2[idx], one_channel=False, unnormalize=False)
        ax.set_title("img2")
    return fig

def validate_path(path):
    if osp.exists(path):
        return path
    elif osp.exists(osp.join(PARENT_PATH, path)):
        return osp.join(PARENT_PATH, path)
    elif osp.exists(osp.join(PROJECT_PATH, path)):
        return osp.join(PROJECT_PATH, path)
    else:
        return FileNotFoundError


def activation_fn(fn_type):
    """Simple switcher for choosing activation functions."""
    if fn_type == 'none':
        def fn(x): return x
    elif fn_type == 'relu':
        fn = nn.ReLU()
    elif fn_type in ['leaky-relu', 'leaky_relu']:
        fn = nn.LeakyReLU()
    elif fn_type == 'tanh':
        fn = nn.Tanh()
    elif fn_type == 'sigmoid':
        fn = nn.Sigmoid()
    elif fn_type == 'softmax':
        fn = nn.Softmax()
    else:
        raise NotImplementedError(
            'Activation function implemented: ' + str(fn_type))

    return fn


def build_topk_mask(x, dim=1, k=2):
    """
    Simple functional version of KWinnersMask/KWinners since
    autograd function apparently not currently exportable by JIT
    Sourced from Jeremy's RSM code
    """
    res = torch.zeros_like(x)
    _, indices = torch.topk(x, k=k, dim=dim, sorted=False)
    return res.scatter(dim, indices, 1)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def xavier_truncated_normal_(tensor, gain=1.0):
    gain = 1.0
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return truncated_normal_(tensor, mean=0.0, std=std)


def initialize_parameters(m, weight_init='xavier_uniform_', bias_init='zeros_'):
    """Initialize nn.Module parameters."""
    if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return

    weight_init_fn = get_initializer_by_name(weight_init)

    if m.weight is not None and weight_init_fn is not None:
        weight_init_fn(m.weight)

    bias_init_fn = get_initializer_by_name(bias_init)

    if m.bias is not None:
        bias_init_fn(m.bias)


def get_initializer_by_name(init_type):
    # Handle custom initializers
    if init_type == 'truncated_normal_':
        return lambda x: truncated_normal_(x, mean=0.0, std=0.03)

    if init_type == 'xavier_truncated_normal_':
        return lambda x: xavier_truncated_normal_(x)

    return getattr(torch.nn.init, init_type, None)


def reduce_max(x, dim=0, keepdim=False):
    """
    Performs `torch.max` over multiple dimensions of `x`
    """
    axes = sorted(dim)
    maxed = x
    for axis in reversed(axes):
        maxed, _ = maxed.max(axis, keepdim)
    return maxed


def get_top_k(x, k, mask_type="pass_through", topk_dim=0, scatter_dim=0):
    """Finds the top k values in a tensor, returns them as a tensor.
    Accepts a tensor as input and returns a tensor of the same size. Values
    in the top k values are preserved or converted to 1, remaining values are
    floored to 0 or -1.
        Example:
            >>> a = torch.tensor([1, 2, 3])
            >>> k = 1
            >>> ans = get_top_k(a, k)
            >>> ans
            torch.tensor([0, 0, 3])
    Args:
        x: (tensor) input.
        k: (int) how many top k examples to return.
        mask_type: (string) Options: ['pass_through', 'hopfield', 'binary']
        topk_dim: (int) Which axis do you want to grab topk over? ie. batch = 0
        scatter_dim: (int) Make it the same as topk_dim to scatter the values
    """

    # Initialize zeros matrix
    zeros = torch.zeros_like(x)

    # find top k vals, indicies
    vals, idx = torch.topk(x, k, dim=topk_dim)

    # Scatter vals onto zeros
    top_ks = zeros.scatter(scatter_dim, idx, vals)

    if mask_type != "pass_through":
        # pass_through does not convert any values.

        if mask_type == "binary":
            # Converts values to 0, 1
            top_ks[top_ks > 0.] = 1.
            top_ks[top_ks < 1.] = 0.

        elif mask_type == "hopfield":
            # Converts values to -1, 1
            top_ks[top_ks >= 0.] = 1.
            top_ks[top_ks < 1.] = -1.

        else:
            raise Exception(
                'Valid options: "pass_through", "hopfield" (-1, 1), or "binary" (0, 1)')

    return top_ks


def square_image_shape_from_1d(filters):
    """
    Make 1d tensor as square as possible. If the length is a prime, the worst case, it will remain 1d.
    Assumes and retains first dimension as batches.
    """
    height = int(math.sqrt(filters))

    while height > 1:
        width_remainder = filters % height
        if width_remainder == 0:
            break
        else:
            height = height - 1

    width = filters // height
    area = height * width
    lost_pixels = filters - area

    shape = [-1, height, width, 1]

    return shape, lost_pixels


def get_padding(kernel_size, stride=1, dilation=1):
    """Calculate symmetric padding for a convolution"""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x, k, s, d):
    """Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution"""
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size, stride=1, dilation=11):
    """Can SAME padding for given args be done statically?"""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same(x, k, s, d=(1, 1), value=0):
    """Dynamically pad input x with 'SAME' padding for conv with specified args"""
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(
        ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    padding = [0, 0, 0, 0]
    if pad_h > 0 or pad_w > 0:
        padding = [pad_w // 2, pad_w - pad_w //
                   2, pad_h // 2, pad_h - pad_h // 2]
        x = F.pad(x, padding, value=value)
    return x, padding


def get_padding_value(padding, kernel_size):
    """Get TF-compatible padding."""
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size):
                # static case, no extra overhead
                padding = get_padding(kernel_size)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size)
    return padding, dynamic


def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Perform a Conv2D operation using SAME padding."""
    stride = (stride, stride) if isinstance(stride, int) else stride

    x, _ = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation,
                    groups=groups)


def conv_transpose2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0),
                          dilation=(1, 1), groups=1, output_shape=None):
    """Perform a ConvTranspose2D operation using SAME padding."""
    stride = (stride, stride) if isinstance(stride, int) else stride

    padded_x, padding = pad_same(x, weight.shape[-2:], stride, dilation)

    # Note: This is kind of hacky way to figure out the correct padding for the
    # transpose operation, depending on the stride
    if stride[0] == 1 and stride[1] == 1:
        x = padded_x
        padding = [padding[0] + padding[1], padding[2] + padding[3]]
    else:
        padding = [padding[0], padding[2]]

    if output_shape is not None:
        out_h = output_shape[2]
        out_w = output_shape[3]

        # Compute output shape
        h = (x.shape[2] - 1) * stride[0] + weight.shape[2] - 2 * padding[0]
        w = (x.shape[3] - 1) * stride[1] + weight.shape[3] - 2 * padding[1]

        output_padding = (out_h - h, out_w - w)

    return F.conv_transpose2d(x, weight=weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, groups=groups)


def max_pool2d_same(x, kernel_size, stride, padding=(0, 0), dilation=(1, 1), ceil_mode=False):
    """Perform MaxPool2D operation using SAME padding."""
    kernel_size = (kernel_size, kernel_size) if isinstance(
        kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride

    x, _ = pad_same(x, kernel_size, stride, value=-float('inf'))
    return F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
