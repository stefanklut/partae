from typing import override

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import grad_norm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import datasets, transforms


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-5,
        optimizer="Adam",
        label_smoothing=0.1,
        freeze_imagenet=False,
        freeze_roberta=False,
    ):
        super(ClassificationModel, self).__init__()
        self.learning_rate = learning_rate

        self.optimizer = optimizer
        self.label_smoothing = label_smoothing
        self.freeze_imagenet = freeze_imagenet
        self.freeze_roberta = freeze_roberta

    @override
    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def get_middle(y):
        return y[:, y.shape[1] // 2]

    def training_step(self, batch, batch_idx):
        B = batch["images"].shape[0]

        _, losses, metrics = self(batch)

        for loss in losses:
            if loss == "loss":
                continue
            self.log(f"train_{loss}", losses[loss], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        self.log("train_loss", sum(losses.values()), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for metric in metrics:
            self.log(f"train_{metric}", metrics[metric], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

        total_loss = sum(losses.values())
        return total_loss

    def validation_step(self, batch, batch_idx):
        B = batch["images"].shape[0]

        _, losses, metrics = self(batch)

        for loss in losses:
            if loss == "loss":
                continue
            self.log(f"val_{loss}", losses[loss], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        self.log("val_loss", sum(losses.values()), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for metric in metrics:
            self.log(f"val_{metric}", metrics[metric], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

    def test_step(self, batch, batch_idx):
        B = batch["images"].shape[0]

        _, losses, metrics = self(batch)

        for loss in losses:
            if loss == "loss":
                continue
            self.log(f"test_{loss}", losses[loss], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        self.log("test_loss", sum(losses.values()), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for metric in metrics:
            self.log(f"test_{metric}", metrics[metric], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

    def configure_optimizers(self):

        if self.optimizer == "Adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")
        return optimizer
