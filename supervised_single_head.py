from argparse import Namespace
from typing import Optional, Dict, Any

import sys
sys.path.append('../')

import numpy as np
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR

from models.macro import unilateral
from lightning import LightningModule


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


class SupervisedLightningModuleSingleHead(LightningModule):
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

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _initialize_model(self):
        mydict = {
                    "mode_out": self.config["hparams"].get("mode_out"),
                    "mode_heads": self.config["hparams"].get("mode_heads"),
                    "arch": self.config["hparams"].get("farch"),
                    "model_path": self.config["hparams"].get("model_path"),
                    "freeze_params": False,
                    "k": self.config["hparams"].get("fine_k"),
                    "per_k": self.config["hparams"].get("fine_per_k"),
                    "dropout": self.config["hparams"].get("dropout", 0.0)
                }
        args = Namespace(**mydict)
        self.model = unilateral(args)
    
    def forward(self, x):
        output = self.model(x)
        return output

    def _step(self, batch, batch_idx):
        img1, label = batch
        output = self(img1)
        loss = self.ce_loss(output, label)
        return loss, output, label

    def _calc_accuracy(self, outputs):
        label_arr = []      # target label
        y_arr = []          # network output
        for out in outputs:
            y, label = out
            _, ind = torch.max(y, 1)
            label_arr.append(label)
            y_arr.append(ind)
        label_arr = torch.cat(label_arr, 0).numpy()
        y_arr = torch.cat(y_arr, 0).numpy()
        acc1 = accuracy_score(label_arr, y_arr)
        return acc1

    def training_step(self, batch, batch_idx):
        loss, output, label = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.training_step_outputs.append((output.detach().cpu(), label.detach().cpu()))
        return loss
    
    def on_train_epoch_end(self) -> None:
        acc = self._calc_accuracy(self.training_step_outputs)
        self.log('train_acc', np.float32(acc))
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        loss, output, label = self._step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append((output.detach().cpu(), label.detach().cpu()))

    def on_validation_epoch_end(self) -> None:
        acc = self._calc_accuracy(self.validation_step_outputs)
        self.log('val_acc', acc)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        loss, output, t = self._step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_step_outputs.append((output.detach().cpu(), t.detach().cpu()))

    def on_test_epoch_end(self) -> None:
        acc = self._calc_accuracy(self.test_step_outputs)
        self.log('test_acc', np.float32(acc))
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        return optimizer
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=self.max_epochs)
        # return [optimizer], [scheduler]
