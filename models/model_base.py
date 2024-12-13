try:
    from typing import override
except ImportError:
    override = lambda x: x

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
    """
    Base class for classification models. Implements the training, validation, and testing steps.
    """

    def __init__(
        self,
        learning_rate=1e-5,
        optimizer="Adam",
        label_smoothing=0.1,
    ):
        super(ClassificationModel, self).__init__()
        self.learning_rate = learning_rate

        self.optimizer = optimizer
        self.label_smoothing = label_smoothing

    def forward(self, x):
        """
        Override this method to define the forward pass of the model.
        """
        raise NotImplementedError

    @staticmethod
    def get_middle(y):
        """
        Get the middle element of a tensor along the second dimension.

        Args:
            y (tensor): Input tensor

        Returns:
            tensor: Middle element of the tensor
        """
        return y[:, y.shape[1] // 2]

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch (dict): Dictionary containing the batch data
            batch_idx (int): Index of the batch

        Returns:
            tensor: Loss value
        """
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
        """
        Validation step of the model.

        Args:
            batch (dict): Dictionary containing the batch data
            batch_idx (int): Index of the batch
        """
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
        """
        Test step of the model.

        Args:
            batch (dict): Dictionary containing the batch data
            batch_idx (int): Index of the batch
        """
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
        """
        Configure the optimizer for the model.
        """

        if self.optimizer == "Adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")
        return optimizer
