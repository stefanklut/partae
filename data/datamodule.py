import sys
from pathlib import Path

import lightning as pl
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from data.dataset import DocumentSeparationDataset


class DocumentSeparationModule(pl.LightningDataModule):
    def __init__(self, train_files, val_files):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DocumentSeparationDataset(self.train_files)
            self.val_dataset = DocumentSeparationDataset(self.val_files)

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)
