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

from models.macro import unilateral
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
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        
        self.weight_decay = self.config['hparams']['weight_decay']
        self.learning_rate = self.config['hparams']['lr']
        # self.warmup_epochs = self.config['hparams']['warmup_epochs']
        # self.max_epochs = self.config['trainer_params']['max_epochs']
            
        self._initialize_model()

        # compute iters per epoch
        self.ce_loss = nn.CrossEntropyLoss()

    def _initialize_model(self):
        mydict = {
                    "mode": self.config["hparams"].get("mode"),
                    "arch": self.config["hparams"].get("arch"),
                    "model_path": self.config["hparams"].get("model_path"),
                    "freeze_params": False,
                    "k": self.config["hparams"].get("k"),
                    "k_percent": self.config["hparams"].get("k_percent"),
                    "dropout": self.config["hparams"].get("dropout", 1.0)
                }
        args = Namespace(**mydict)
        self.model = unilateral(args)
    
    def forward(self, x):
        output = self.model(x)
        return output

    def _eval_step(self, batch, batch_idx):
        img1, label = batch
        output = self(img1)
        loss_sim = self.ce_loss(output, label)
        loss = loss_sim
        return loss, output, label

    def _calc_accuracy(self, outputs, train_outputs=False):
        label_arr = []
        output_arr = []
        for out in outputs:
            if train_outputs:
                output, label = out['output']
            else:
                output, label = out
            _, ind = torch.max(output, 1)
            label_arr.append(label)
            output_arr.append(ind)
        label_arr = torch.cat(label_arr, 0).numpy()
        output_arr = torch.cat(output_arr, 0).numpy()
        acc1 = accuracy_score(label_arr, output_arr)
        return acc1

    def training_step(self, batch, batch_idx):
        loss, output, label = self._eval_step(batch, batch_idx)

        self.log("train loss", loss, on_step=True, on_epoch=False)
        return {'loss':loss, 'output':(output.detach().cpu(), label.detach().cpu())}
    
    def training_epoch_end(self, outputs: Any) -> None:
        acc = self._calc_accuracy(outputs, train_outputs=True)
        self.log('train_acc', acc)

    def validation_step(self, batch, batch_idx):
        loss, output, t = self._eval_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, 
                 on_epoch=True, sync_dist=True)
        return (output.detach().cpu(), t.detach().cpu())

    def validation_epoch_end(self, outputs: Any) -> None:
        acc = self._calc_accuracy(outputs)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        loss, output, t = self._eval_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, 
                 on_epoch=True, sync_dist=True)
        return (output.detach().cpu(), t.detach().cpu())

    def test_epoch_end(self, outputs: Any) -> None:
        acc = self._calc_accuracy(outputs)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        return optimizer
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=self.max_epochs)
        # return [optimizer], [scheduler]
