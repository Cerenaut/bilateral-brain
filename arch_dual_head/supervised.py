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
from lightning import LightningModule
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

        self.weight_decay = self.config['hparams']['weight_decay']
        self.learning_rate = self.config['hparams']['lr']
        # self.warmup_epochs = self.config['hparams']['warmup_epochs']
        # self.max_epochs = self.config['trainer_params']['max_epochs']

        # TODO add config param to specify model or ensemble model
        self._initialize_model()
        # self._initialize_ensemble_model()
        self.ce_loss = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.eval_step_outputs = []

 
    def _initialize_model(self):
        macro_arch = self.config["hparams"].get("macro_arch", None)
        if macro_arch == 'bilateral' or macro_arch == None:
            print("----- **BILATERAL** macro-architecture")

            mydict = {
                "mode": self.config["hparams"]["mode"],
                "farch": self.config["hparams"].get("farch", None),
                "carch": self.config["hparams"].get("carch", None),
                "fmodel_path": self.config["hparams"].get("model_path_fine"),
                "cmodel_path": self.config["hparams"].get("model_path_coarse"),
                "ffreeze_params": self.config["hparams"].get("ffreeze"),
                "cfreeze_params": self.config["hparams"].get("cfreeze"),
                "fine_k": self.config["hparams"].get("fine_k"),
                "fine_per_k": self.config["hparams"].get("fine_per_k"),
                "coarse_k": self.config["hparams"].get("coarse_k"),
                "coarse_per_k": self.config["hparams"].get("coarse_per_k"),
                "dropout": self.config["hparams"].get("dropout", 1.0),
                }
            args = Namespace(**mydict)
            self.model = bilateral(args)
        else:
            print("----- **UNILATERAL** macro-architecture")

            mydict = {
                "mode": self.config["hparams"]["mode"],
                "arch": self.config["hparams"].get("farch", None),
                "model_path": self.config["hparams"].get("model_path_fine"),
                "freeze_params": self.config["hparams"].get("ffreeze"),
                "k": self.config["hparams"].get("fine_k"),
                "per_k": self.config["hparams"].get("fine_per_k"),
                "dropout": self.config["hparams"].get("dropout", 0.0),
                }
            args = Namespace(**mydict)
            self.model = unilateral(args)
    
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
    
    def forward(self, x):
        fine, coarse = self.model(x)
        return fine, coarse

    def _step(self, batch):
        '''
        suffixes:
        t = target (the label)
        y = output
        '''
        img1, finey, coarsey = batch['image'], batch['fine'], batch['coarse']
        finet, coarset = self(img1)
        loss_fine = self.ce_loss(finet, finey) 
        loss_coarse = self.ce_loss(coarset, coarsey)
        return (loss_fine, loss_coarse), (finey, coarsey), (finet, coarset)

    def _calc_accuracy(self, outputs) -> Any:
        y_fine_arr = []     # network output
        t_fine_arr = []     # target labels
        y_coarse_arr = []
        t_coarse_arr = []
        for out in outputs:
            fine_out, coarse_out = out
            y_fine, t_fine = fine_out
            y_coarse, t_coarse = coarse_out
            _, fine_ind = torch.max(y_fine, 1)
            _, coarse_ind = torch.max(y_coarse, 1)
            y_fine_arr.append(fine_ind)
            t_fine_arr.append(t_fine)
            y_coarse_arr.append(coarse_ind)
            t_coarse_arr.append(t_coarse)
        y_fine_arr = torch.cat(y_fine_arr, 0).numpy()
        y_coarse_arr = torch.cat(y_coarse_arr, 0).numpy()
        t_fine_arr = torch.cat(t_fine_arr, 0).numpy()
        t_coarse_arr = torch.cat(t_coarse_arr, 0).numpy()
        acc_fine = accuracy_score(t_fine_arr, y_fine_arr)
        acc_coarse = accuracy_score(t_coarse_arr, y_coarse_arr)
        return acc_fine, acc_coarse

    def training_step(self, batch, batch_idx):
        (loss_fine, loss_coarse), (finey, coarsey), (finet, coarset) = self._step(batch)
        loss = loss_fine + loss_coarse

        self.log('train narrow loss', loss_fine)
        self.log('train broad loss', loss_coarse)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)        
    
        step_output = (finet.detach().cpu(), finey.detach().cpu()), (coarset.detach().cpu(), coarsey.detach().cpu())
        self.training_step_outputs.append(step_output)
        return loss
    
    def on_train_epoch_end(self) -> None:
        acc_fine, acc_coarse = self._calc_accuracy(self.training_step_outputs)
        self.log('train_acc_fine', acc_fine)
        self.log('train_acc_coarse', acc_coarse)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        (loss_fine, loss_coarse), (finey, coarsey), (finet, coarset) = self._step(batch)
        loss = loss_fine + loss_coarse

        self.log(f'val_loss_fine', loss_fine)
        self.log(f'val_loss_coarse', loss_coarse)
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        step_output = (finey.detach().cpu(), finet.detach().cpu()), (coarsey.detach().cpu(), coarset.detach().cpu())
        self.eval_step_outputs.append(step_output)

    def on_validation_epoch_end(self) -> None:
        acc_fine, acc_coarse = self._calc_accuracy(self.eval_step_outputs)
        self.log(f'val_acc_fine', acc_fine)
        self.log(f'val_acc_coarse', acc_coarse)
        self.eval_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        (loss_fine, loss_coarse), (finey, coarsey), (finet, coarset) = self._step(batch)
        loss = loss_fine + loss_coarse

        self.log(f'test_loss_fine', loss_fine)
        self.log(f'test_loss_coarse', loss_coarse)
        self.log(f'test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        step_output = (finey.detach().cpu(), finet.detach().cpu()), (coarsey.detach().cpu(), coarset.detach().cpu())
        self.eval_step_outputs.append(step_output)

    def on_test_epoch_end(self) -> None:
        acc_fine, acc_coarse = self._calc_accuracy(self.eval_step_outputs)
        self.log(f'test_acc_fine', acc_fine)
        self.log(f'test_acc_coarse', acc_coarse)
        self.eval_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
        # return [optimizer], [scheduler]