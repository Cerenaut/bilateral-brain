from argparse import Namespace
from typing import Optional, Dict, Any
import sys
sys.path.append('../')

import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from models.macro import bilateral, unilateral
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from sklearn.metrics import accuracy_score
from utils import plot_grad_flow, plot_grad_flow_v2, \
        inverse_normalize, matplotlib_imshow


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
        # self.gpus = self.config['trainer_params']['gpus']
        # self.batch_size = self.config['hparams']['batch_size']

        self.optim = self.config['hparams']['optimizer']
        self.weight_decay = self.config['hparams']['weight_decay']

        self.learning_rate = self.config['hparams']['lr']
        # self.warmup_epochs = self.config['hparams']['warmup_epochs']
        # self.max_epochs = self.config['trainer_params']['max_epochs']

        # TODO add config param to specify model or ensemble model
        self._initialize_model()
        # self._initialize_ensemble_model()
        self.ce_loss = nn.CrossEntropyLoss()

    """
    Get hparam value from config
    Specify if it must be explicit with `is_must` and if not, use to `default`
    """    
    def _get_hparam(self, hparam_key, is_must=False, default=None):
        if hparam_key in self.config["hparams"]:
            return self.config["hparams"][hparam_key]
        elif is_must:
            raise ValueError("hparam_key {} not found in config".format(hparam_key))
        else:
            return default
        
    def _initialize_model(self):
        mydict = {
            "mode": self._get_hparam("mode", is_must=True),
            "carch": self._get_hparam("carch", is_must=True),
            "farch": self._get_hparam("harch", is_must=True),
            "cmodel_path": self._get_hparam("model_path_coarse"),
            "fmodel_path": self._get_hparam("fmodel_path_fine"),
            "cfreeze_params": True,
            "ffreeze_params": True,
            "narrow_k": self._get_hparam("narrow_k"),
            "narrow_per_k": self._get_hparam("narrow_k_percent"),
            "broad_k": self._get_hparam("broad_k"),
            "broad_per_k": self._get_hparam("broad_k_percent"),
            "dropout": self._get_hparam("dropout", is_must=False, default=1.0),
            }
        args = Namespace(**mydict)
        self.model = bilateral(args)
    
    def _initialize_ensemble_model(self):
        self.k = None
        self.k_percent = None
        mydict = {
                    "mode": self.mode, 
                    "arch": self.arch,
                    "k": self.k, 
                    "k_percent":self.k_percent
                }
        args = Namespace(**mydict)
        self.model = unilateral(args)
    
    def _step(self, img1):
        narrow, broad = self.model(img1)
        return narrow, broad

    def training_step(self, batch, batch_idx):
        img1, finey, coarsey = batch['image'], batch['fine'], batch['coarse']
        narrowt, broadt = self._step(img1)
        loss_narr = self.ce_loss(narrowt, finey) 
        loss_broad = self.ce_loss(broadt, coarsey)
        loss = loss_narr + loss_broad

        self.log('train narrow loss', loss_narr)
        self.log('train broad loss', loss_broad)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, sync_dist=True)        
        return {'loss':loss, 
                'output':((narrowt.detach().cpu(), finey.detach().cpu()), 
                        (broadt.detach().cpu(), coarsey.detach().cpu()))}
    
    def training_epoch_end(self, output: Any) -> None:
        acc_narr, acc_broad = self._epoch_end(output)
        self.log('train_acc_narr', acc_narr)
        self.log('train_acc_broad', acc_broad)

    def _epoch_end(self, output: Any) -> Any:
        narrowts = []
        fineys = []
        broadts = []
        coarseys = []
        for out in output:
            if isinstance(out, dict):
                narrow, broad = out['output']
            else:
                narrow, broad = out
            narrowt, finey = narrow
            broadt, coarsey = broad
            _, narrowind = torch.max(narrowt, 1)
            _, broadind = torch.max(broadt, 1)
            narrowts.append(narrowind)
            fineys.append(finey)
            broadts.append(broadind)
            coarseys.append(coarsey)
        narrowts = torch.cat(narrowts, 0).numpy()
        broadts = torch.cat(broadts, 0).numpy()
        fineys = torch.cat(fineys, 0).numpy()
        coarseys = torch.cat(coarseys, 0).numpy()
        acc_narr = accuracy_score(fineys, narrowts)
        acc_broad = accuracy_score(coarseys, broadts)
        return acc_narr, acc_broad
    
    def validation_step(self, batch, batch_idx):
        img1, finey, coarsey = batch['image'], batch['fine'], batch['coarse']
        narrowt, broadt = self._step(img1)
        loss_narr = self.ce_loss(narrowt, finey) 
        loss_broad = self.ce_loss(broadt, coarsey)
        loss = loss_narr + loss_broad

        self.log('val narrow loss', loss_narr)
        self.log('val broad loss', loss_broad)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, sync_dist=True)

        return ((narrowt.detach().cpu(), finey.detach().cpu()), 
                        (broadt.detach().cpu(), coarsey.detach().cpu()))

    def validation_epoch_end(self, output: Any) -> None:
        acc_narr, acc_broad = self._epoch_end(output)
        self.log('val_acc_narr', acc_narr)
        self.log('val_acc_broad', acc_broad)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
        # return [optimizer], [scheduler]