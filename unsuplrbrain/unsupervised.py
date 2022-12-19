from typing import Optional, Dict, Any

import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from model import SparseAutoencoder
from pytorch_lightning import LightningModule

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from utils import plot_grad_flow, plot_grad_flow_v2, \
        inverse_normalize, matplotlib_imshow


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(
            torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class UnsupervisedLightningModule(LightningModule):
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
        k = self.config['hparams']['k']
        k_percent = self.config['hparams']['per_k']
        latent_dim = self.config['hparams']['latent_dim']

        self.model = SparseAutoencoder(in_channels=3, 
                                latent_dim=latent_dim,
                                k=k,
                                k_percent=k_percent)

        # compute iters per epoch
        self.loss = nn.MSELoss(reduction='sum')

    def training_step(self, batch, batch_idx):
        img1, y = batch
        encode1, decode1 = self.model(img1)
        loss_sim = 0 #self.nt_xent_loss(encode1, encode2)
        loss_mse = self.loss(decode1, img1)
        loss=loss_mse + loss_sim
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
        return loss

    def validation_step(self, batch, batch_idx):
        img1, y = batch
        encode1, decode1 = self.model(img1)
        loss_sim = 0 #self.nt_xent_loss(encode1, encode2)
        loss_mse = self.loss(decode1, img1)
        loss = loss_mse + loss_sim

        self.log('val mse loss', loss_mse)

        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, sync_dist=True)

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