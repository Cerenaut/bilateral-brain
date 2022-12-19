from typing import Optional, Dict, Any
import sys
sys.path.append('../')

import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from unsuplrbrain.model import SparseAutoencoder
from pytorch_lightning import LightningModule

from sklearn.metrics import accuracy_score
from utils import plot_grad_flow, plot_grad_flow_v2


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
        if 'k' in self.config['hparams']:
            self.k = self.config['hparams']['k']
        if 'per_k' in self.config['hparams']:
            self.per_k = self.config['hparams']['per_k']
        self.latent_dim = self.config['hparams']['latent_dim']
        self.model = SparseAutoencoder(in_channels=3,
                                        latent_dim=self.latent_dim)

        # compute iters per epoch
        self.loss = nn.MSELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        img1, y = batch
        output, decode1 = self.model(img1)
        # loss_sim = self.ce_loss(output, y) 
        loss_mse = self.loss(decode1, img1)
        loss=loss_mse #+ loss_sim
        vutils.save_image(decode1,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}.png",
                          normalize=True,
                          scale_each=True,
                          nrow=8)
        vutils.save_image(img1,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"{self.logger.name}.png",
                          normalize=True,
                          scale_each=True,
                          nrow=8)
        self.log('train mse loss', loss_mse)
        # self.log('train sim loss', loss_sim)
        self.log("train loss", loss, on_step=True, on_epoch=False)
        return {'loss':loss, 'output':(output.detach().cpu(), y.detach().cpu())}
    
    # def training_epoch_end(self, output: Any) -> None:
    #     targets = []
    #     outputs = []
    #     for out in output:
    #         labels, target = out['output']
    #         _, ind = torch.max(labels, 1)
    #         targets.append(target)
    #         outputs.append(ind)
    #     targets = torch.cat(targets, 0).numpy()
    #     outputs = torch.cat(outputs, 0).numpy()
    #     acc1 = accuracy_score(targets, outputs)
    #     self.log('train_acc', acc1)

    def validation_step(self, batch, batch_idx):
        img1, y = batch
        output, decode1 = self.model(img1)
        # loss_sim = self.ce_loss(output, y)
        loss_mse = self.loss(decode1, img1)
        loss = loss_mse #+ loss_sim

        self.log('val mse loss', loss_mse)
        # self.log('val sim loss', loss_sim)

        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        return {'loss':loss, 'output':(output.detach().cpu(), y.detach().cpu())}
    
    # def validation_epoch_end(self, output: Any) -> None:
    #     targets = []
    #     outputs = []
    #     for out in output:
    #         labels, target = out
    #         _, ind = torch.max(labels, 1)
    #         targets.append(target)
    #         outputs.append(ind)
    #     targets = torch.cat(targets, 0).numpy()
    #     outputs = torch.cat(outputs, 0).numpy()
    #     acc1 = accuracy_score(targets, outputs)
    #     self.log('val_acc', acc1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.learning_rate, weight_decay=self.weight_decay)
        return [optimizer]
    
    def backward(
        self, loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], *args, **kwargs
    ) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your
        own implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).
            optimizer: Current optimizer being used. ``None`` if using manual optimization.
            optimizer_idx: Index of the current optimizer being used. ``None`` if using manual optimization.

        Example::

            def backward(self, loss, optimizer, optimizer_idx):
                loss.backward()
        """
        loss.backward(*args, **kwargs)
        self.logger.experiment.add_figure('grad v1 figure', 
                                plot_grad_flow(self.model.named_parameters()), global_step=self.current_epoch)
        self.logger.experiment.add_figure('grad v2 figure', 
                                plot_grad_flow_v2(self.model.named_parameters()), global_step=self.current_epoch)