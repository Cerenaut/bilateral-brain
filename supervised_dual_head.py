from argparse import Namespace
from typing import Optional, Dict, Any

import torch
from torch import nn
from lightning import LightningModule
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../')

from models.macro import bilateral, unilateral, ensemble
from utils import setup_logger

logger = setup_logger(__name__)


class SupervisedLightningModuleDualHead(LightningModule):
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

        self._initialize_model()

        self.ce_loss = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.eval_step_outputs = []

 
    def _initialize_model(self):
        '''
        Assumes that if it's an ensemble model, that all models have identical hparams
        '''

        mydict = {
            "mode_out": self.config["hparams"]["mode_out"],
            "mode_heads": self.config["hparams"]["mode_heads"],
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

        mode_hemis = self.config["hparams"].get("mode_hemis", None)
        logger.info("----- ** {model_hemis} ** -------- hemisphere configuration")
        if mode_hemis == 'bilateral':
            self.model = bilateral(args)    # mode_heads ignored, assumes 'both'
        elif mode_hemis == 'unilateral':
            self.model = unilateral(args)
        elif mode_hemis == 'ensemble':  
            self.model = ensemble(args)     # modes will be ignored (because ensemble assumes values)
        else:
            raise ValueError(f"Invalid mode_hemis specification: {mode_hemis}")
    
    def forward(self, x):
        fine, coarse = self.model(x)
        return fine, coarse

    def _step(self, batch):
        '''
        suffixes:
        t = target (the label)
        y = output
        '''
        img1, t_fine, t_coarse = batch['image'], batch['fine'], batch['coarse']
        y_fine, y_coarse = self(img1)
        loss_fine = self.ce_loss(y_fine, t_fine) 
        loss_coarse = self.ce_loss(y_coarse, t_coarse)
        return (loss_fine, loss_coarse), (y_fine, y_coarse), (t_fine, t_coarse)

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
        loss, y, t = self._step(batch)

        loss_fine, loss_coarse = loss
        y_fine, y_coarse = y
        t_fine, t_coarse = t

        loss = loss_fine + loss_coarse

        self.log('train_narrow_loss', loss_fine)
        self.log('train_broad_loss', loss_coarse)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)        
    
        step_output = (y_fine.detach().cpu(), t_fine.detach().cpu()), (y_coarse.detach().cpu(), t_coarse.detach().cpu())
        self.training_step_outputs.append(step_output)
        return loss
    
    def on_train_epoch_end(self) -> None:
        acc_fine, acc_coarse = self._calc_accuracy(self.training_step_outputs)
        self.log('train_acc_fine', acc_fine)
        self.log('train_acc_coarse', acc_coarse)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        loss, y, t = self._step(batch)
        
        loss_fine, loss_coarse = loss
        y_fine, y_coarse = y
        t_fine, t_coarse = t

        loss = loss_fine + loss_coarse

        self.log(f'val_loss_fine', loss_fine)
        self.log(f'val_loss_coarse', loss_coarse)
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        step_output = (y_fine.detach().cpu(), t_fine.detach().cpu()), (y_coarse.detach().cpu(), t_coarse.detach().cpu())
        self.eval_step_outputs.append(step_output)

    def on_validation_epoch_end(self) -> None:
        acc_fine, acc_coarse = self._calc_accuracy(self.eval_step_outputs)
        acc = acc_fine + acc_coarse
        self.log(f'val_acc', acc)
        self.log(f'val_acc_fine', acc_fine)
        self.log(f'val_acc_coarse', acc_coarse)
        self.eval_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        loss, y, t = self._step(batch)
        
        loss_fine, loss_coarse = loss
        y_fine, y_coarse = y
        t_fine, t_coarse = t

        loss = loss_fine + loss_coarse

        self.log(f'test_loss_fine', loss_fine)
        self.log(f'test_loss_coarse', loss_coarse)
        self.log(f'test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        step_output = (y_fine.detach().cpu(), t_fine.detach().cpu()), (y_coarse.detach().cpu(), t_coarse.detach().cpu())
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