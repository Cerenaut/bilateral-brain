from argparse import Namespace
from typing import Optional, Dict, Any

import sys
sys.path.append('../')

import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR

from models.resnet import resnet18, resnet34
from models.resnet_v0 import resnet9, invresnet9
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from sklearn.metrics import accuracy_score

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class SupervisedLightningModule(LightningModule):
    def __init__(
        self,
        config: Optional[Dict],
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        
        self.batch_size = self.config['hparams']['batch_size']
        self.optim = self.config['hparams']['optimizer']
        self.weight_decay = self.config['hparams']['weight_decay']

        self.learning_rate = self.config['hparams']['lr']
        self.warmup_epochs = self.config['hparams']['warmup_epochs']
        self.max_epochs = self.config['trainer_params']['max_epochs']

        self.k = self.config['hparams']['k']
        self.k_percent = self.config['hparams']['per_k']
        if 'model_path' in self.config['hparams'].keys():
            self.model_path = self.config['hparams']['model_path']
        self._initialize_model()

        # compute iters per epoch
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _initialize_model(self):
        if self.k == 0:
            self.k = None
        if self.k_percent == 0:
            self.k_percent = None
        mydict = {
                    "mode": "narrow", 
                    "k": self.k, 
                    "k_percent": self.k_percent
                }
        args = Namespace(**mydict)
        self.model = resnet9(args)
    
    def _load_model(self):    
        model_dict = torch.load(self.model_path)['state_dict']
        model_dict = {k.replace('model.encoder.', ''):v for k,v in model_dict.items() if 'encoder' in k}
        self.model.load_state_dict(model_dict)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        output = self.model(x)
        return output
        
    def training_step(self, batch, batch_idx):
        img1, y = batch
        output = self(img1)
        loss_sim = self.ce_loss(output, y)
        loss=loss_sim
        self.log("train loss", loss, on_step=True, on_epoch=False)
        return {'loss':loss, 'output':(output.detach().cpu(), y.detach().cpu())}
    
    def training_epoch_end(self, output: Any) -> None:
        targets = []
        outputs = []
        for out in output:
            labels, target = out['output']
            _, ind = torch.max(labels, 1)
            targets.append(target)
            outputs.append(ind)
        targets = torch.cat(targets, 0).numpy()
        outputs = torch.cat(outputs, 0).numpy()
        acc1 = accuracy_score(targets, outputs)
        self.log('train_acc', acc1)

    def validation_step(self, batch, batch_idx):
        img1, y = batch
        output = self(img1)
        loss_sim = self.ce_loss(output, y)
        loss = loss_sim

        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        return (output.detach().cpu(), y.detach().cpu())
    
    def validation_epoch_end(self, output: Any) -> None:
        targets = []
        outputs = []
        for out in output:
            labels, target = out
            _, ind = torch.max(labels, 1)
            targets.append(target)
            outputs.append(ind)
        targets = torch.cat(targets, 0).numpy()
        outputs = torch.cat(outputs, 0).numpy()
        acc1 = accuracy_score(targets, outputs)
        self.log('val_acc', acc1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        return optimizer
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=self.max_epochs)
        # return [optimizer], [scheduler]