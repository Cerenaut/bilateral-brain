"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

def check_list():
    return ['narrow', 'broad', 'both', 'feature']


# from pygln import gln

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, mode: str = 'both'):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mode = mode
        if self.mode not in check_list():
            raise Exception('Mode of training does not match')
        if self.mode == 'narrow':
            self.fc = nn.Linear(in_features=512, out_features=100, bias=True)
        elif self.mode == 'broad':
            self.fc = nn.Linear(in_features=512, out_features=20, bias=True)
        elif self.mode == 'both':
            self.ffc = nn.Linear(in_features=512, out_features=100, bias=True)
            self.cfc = nn.Linear(in_features=512, out_features=20, bias=True)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers (by layer I didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        if self.mode == 'narrow' or self.mode == 'broad':
            return self.fc(output)
        elif self.mode == 'both':
            return self.ffc(output), self.cfc(output)
        elif self.mode =='feature':
            return output

def resnet18(mode):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], mode=mode)

def resnet34(mode):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], mode=mode)

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class Net(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, mode: str = 'both'):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,  padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.mode = mode
        if self.mode not in check_list():
            raise Exception('Mode of training does not match')
        if self.mode == 'narrow':
            self.fc = nn.Linear(in_features=512, out_features=100, bias=True)
        elif self.mode == 'broad':
            self.fc = nn.Linear(in_features=512, out_features=20, bias=True)
        elif self.mode == 'both':
            self.ffc = nn.Linear(in_features=512, out_features=100, bias=True)
            self.cfc = nn.Linear(in_features=512, out_features=20, bias=True)

    def forward(self, x):
        out = self.conv(x)
        if self.mode == 'narrow' or self.mode == 'broad':
            return self.fc(out)
        elif self.mode == 'both':
            return self.ffc(out), self.cfc(out)
        elif self.mode =='feature':
            return out

def resnet9(mode):
    """ return a ResNet 18 object
    """
    return Net(mode)

def load_bicam_model(ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)
    # ['state_dict']
    model_dict = {k.replace('model.', ''):v for k,v in sdict.items() if 'conv' in k}
    return model_dict

def load_unicam_model(ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)
    # ['state_dict']
    model_dict = {k.replace('model.', ''):v for k,v in sdict.items() if 'conv' in k}
    return model_dict

def freeze_params(model):
    """[summary]

    Args:
        model ([type]): [description]
    """
    for param in model.parameters():
        param.requires_grad = False

class BicamNet(nn.Module):
    """
    A Bicameral Residual network.
    """
    def __init__(self,
                    arch: str, 
                    mode: str = 'feature',
                    cmodel_path = None,
                    fmodel_path = None,
                    cfreeze_params: bool = False,
                    ffreeze_params: bool = False,):
        super(BicamNet, self).__init__()
        self.broad = arch(mode=mode)
        self.narrow = arch(mode=mode)
        if cmodel_path is not None:
            self.broad.load_state_dict(load_bicam_model(cmodel_path))
        if fmodel_path is not None:
            self.narrow.load_state_dict(load_bicam_model(fmodel_path))
        if cfreeze_params:
            freeze_params(self.broad)
        if ffreeze_params:
            freeze_params(self.narrow)

        # add the 
        self.fcombiner = nn.Sequential(
                            nn.Linear(1024, 100)
                            )
        self.ccombiner = nn.Sequential(
                            nn.Linear(1024,  20)
                            )

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]
        """
        fembed = self.narrow(x)
        cembed = self.broad(x)
        embed = torch.cat([fembed, cembed], axis=1)
        foutput = self.fcombiner(embed)
        coutput = self.ccombiner(embed)
        return foutput, coutput

def bicameral(args):
    """[summary]
    """
    return BicamNet(args.arch, 
                    args.mode, 
                    args.cmodel_path, args.fmodel_path, 
                    args.cfreeze_params, args.ffreeze_params)

def unicameral(args):
    """[summary]
    """
    return args.arch(args.mode)