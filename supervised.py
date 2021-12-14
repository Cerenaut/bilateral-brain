from typing import Optional, Dict, Any

import sys
sys.path.append('../')
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer


from model import MLP, ResNet9, SparseAutoencoder
from unsuplrbrain.model import ResNet9Enc
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from sklearn.metrics import accuracy_score


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
        self.gpus = self.config['trainer_params']['gpus']
        self.batch_size = self.config['hparams']['batch_size']

        self.optim = self.config['hparams']['optimizer']
        self.weight_decay = self.config['hparams']['weight_decay']

        self.learning_rate = self.config['hparams']['lr']
        self.warmup_epochs = self.config['hparams']['warmup_epochs']
        self.max_epochs = self.config['trainer_params']['max_epochs']

        self.k = self.config['hparams']['k']
        self.k_percent = self.config['hparams']['per_k']
        self.model_path = self.config['hparams']['model_path']
        self._initialize_model()

        # compute iters per epoch
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _initialize_model(self):
        if self.k == 0:
            self.k = None
        if self.k_percent == 0:
            self.k_percent = None

        self.model = SparseAutoencoder(in_channels=3, 
                                    num_classes=self.config['hparams']['num_classes'])
    
    def forward(self, x):
        output = self.model(x)
        return output
        
    def training_step(self, batch, batch_idx):
        img1, y = batch
        output = self.model(img1)
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
        params = []
        params += list(self.model.parameters())
        optimizer = torch.optim.Adam(params, 
            lr=self.learning_rate, weight_decay=self.weight_decay)
        steps_per_epoch = (len(self.train_dataloader()) \
                            // self.trainer.accumulate_grad_batches)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1,
                                max_epochs=self.max_epochs)
        return [optimizer], [scheduler]