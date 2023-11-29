"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from utils import setup_logger
import torchvision.models as models

# from pygln import gln


logger = setup_logger(__name__)

''' VGG as a feature extractor - strips out the classifier, and returns the features '''
class VGG11(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.vgg = models.vgg11(pretrained=False)

        self.num_features = 512 * 7 * 7

        # remove the classifier
        self.vgg.classifier = nn.Identity()

        # # replace the classifier
        # self.vgg.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, self.num_features),
        #     nn.ReLU(True),
        # )

    def forward(self, x):
        return self.vgg(x)

def vgg11(args):
    """ return a vgg11 object
    """
    logger.warning("This implementation of vgg does not accept parameters i.e. no sparsity.")
    return VGG11()

