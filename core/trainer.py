import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import datasets, transforms


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(ClassificationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def split_input(self, batch):
        y = batch["targets"]
        y = y.type(torch.int64)
        y = y.view(-1)
        del batch["targets"]
        x = batch
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.split_input(batch)

        y_hat = self.model(x)
        y_hat = y_hat.view(-1, 2)

        loss = F.cross_entropy(y_hat, y)
        acc = self.train_accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.split_input(batch)

        y_hat = self.model(x)
        y_hat = y_hat.view(-1, 2)

        loss = F.cross_entropy(y_hat, y)
        acc = self.val_accuracy(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = self.split_input(batch)

        y_hat = self.model(x)
        y_hat = y_hat.view(-1, 2)

        loss = F.cross_entropy(y_hat, y)
        acc = self.test_accuracy(y_hat, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
