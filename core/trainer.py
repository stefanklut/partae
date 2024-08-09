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
        model,
        learning_rate=1e-5,
        optimizer="Adam",
        label_smoothing=0.1,
        freeze_imagenet=False,
        freeze_roberta=False,
    ):
        super(ClassificationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.optimizer = optimizer
        self.label_smoothing = label_smoothing
        self.freeze_imagenet = freeze_imagenet
        self.freeze_roberta = freeze_roberta

    def forward(self, x):
        return self.model(x)

    def get_middle_scan(self, y):
        return y[:, y.shape[1] // 2]

    def training_step(self, batch, batch_idx):
        B = batch["images"].shape[0]

        _, losses, metrics = self.model(batch)

        for loss in losses:
            self.log(f"train_{loss}", losses[loss], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for metric in metrics:
            self.log(f"train_{metric}", metrics[metric], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

        total_loss = sum(losses.values())
        return total_loss

    def validation_step(self, batch, batch_idx):
        B = batch["images"].shape[0]

        _, losses, metrics = self.model(batch)

        for loss in losses:
            self.log(f"val_{loss}", losses[loss], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for metric in metrics:
            self.log(f"val_{metric}", metrics[metric], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

    def test_step(self, batch, batch_idx):
        B = batch["images"].shape[0]

        _, losses, metrics = self.model(batch)

        for loss in losses:
            self.log(f"test_{loss}", losses[loss], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for metric in metrics:
            self.log(f"test_{metric}", metrics[metric], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

    def configure_optimizers(self):
        param_groups = []
        for name, param in self.model.named_parameters():
            if "imagenet" in name:
                if self.freeze_imagenet:
                    param.requires_grad = False
                    lr = 0
                lr = self.learning_rate * 0.01
            if "roberta" in name:
                if self.freeze_roberta:
                    param.requires_grad = False
                    lr = 0
                lr = self.learning_rate * 0.01
            else:
                lr = self.learning_rate
            param_groups.append({"params": param, "lr": lr})

        if self.optimizer == "Adam":
            optimizer = optim.Adam(param_groups, lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(param_groups, lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")
        return optimizer
