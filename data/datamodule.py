import itertools
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import pytorch_lightning as pl
from natsort import natsorted
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from data.convert_xlsx import link_with_paths
from data.dataloader import collate_fn
from data.dataset import DocumentSeparationDataset
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())


def split_training_paths(
    training_paths: Sequence[Path], split_ratio: float = 0.9, seed: int = 42
) -> tuple[Sequence[Path], Sequence[Path]]:
    """
    Split training paths into training and validation paths while keeping files from the same folder together. And keeping the order of the files.

    Args:
        training_paths (Sequence[Path]): Paths to the training files
        split_ratio (float): Ratio to split the training paths. Default is 0.8

    Returns:
        Tuple[Sequence[Path], Sequence[Path]]: Tuple of training and validation paths
    """
    training_paths = natsorted(training_paths)
    # Group the paths by parent folder
    grouped_training_paths = [list(group) for _, group in itertools.groupby(training_paths, key=lambda x: x.parent)]

    # Shuffle the groups
    random.seed(seed)
    random.shuffle(grouped_training_paths)
    random.seed()

    # Split the groups
    split_idx = int(len(grouped_training_paths) * split_ratio)
    split_training_paths = list(itertools.chain.from_iterable(grouped_training_paths[:split_idx]))
    split_val_paths = list(itertools.chain.from_iterable(grouped_training_paths[split_idx:]))

    if len(split_training_paths) == 0:
        raise ValueError("Split did not have enough directories for training")
    if len(split_val_paths) == 0:
        raise ValueError("Split did not have enough directories for validation")

    return split_training_paths, split_val_paths


class DocumentSeparationModuleXLSX(pl.LightningDataModule):
    def __init__(
        self,
        training_paths: Sequence[Path],
        val_paths: Optional[Sequence[Path]],
        xlsx_file: Path,
        transform=None,
        batch_size: int = 8,
        number_of_images: int = 3,
        num_workers: int = 8,
        sample_same_inventory: bool = False,
        wrap_round: bool = False,
        prob_shuffle_document: float = 0.0,
        prob_randomize_document_order: float = 0.0,
        prob_random_scan_insert: float = 0.0,
        split_ratio: float = 0.8,
    ):
        super().__init__()

        if val_paths is None:
            # split training paths into training and validation paths
            training_paths, val_paths = split_training_paths(training_paths, split_ratio=split_ratio, seed=101)
        else:
            val_paths = natsorted(val_paths)

        self.xlsx_file = xlsx_file
        self.transform = transform
        self.batch_size = batch_size
        self.number_of_images = number_of_images
        self.num_workers = num_workers

        self.sample_same_inventory = sample_same_inventory
        self.wrap_round = wrap_round

        self.prob_shuffle_document = prob_shuffle_document
        self.prob_randomize_document_order = prob_randomize_document_order
        self.prob_random_scan_insert = prob_random_scan_insert

        self.training_paths = link_with_paths(self.xlsx_file, training_paths)
        self.val_paths = link_with_paths(self.xlsx_file, val_paths)

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DocumentSeparationDataset(
                self.training_paths,
                number_of_images=self.number_of_images,
                transform=self.transform,
                sample_same_inventory=self.sample_same_inventory,
                wrap_round=self.wrap_round,
                prob_shuffle_document=self.prob_shuffle_document,
                prob_randomize_document_order=self.prob_randomize_document_order,
                prob_random_scan_insert=self.prob_random_scan_insert,
            )
            self.val_dataset = DocumentSeparationDataset(
                self.val_paths,
                number_of_images=self.number_of_images,
                transform=self.transform,
                sample_same_inventory=True,
                wrap_round=False,
                prob_shuffle_document=0.0,
                prob_randomize_document_order=0.0,
                prob_random_scan_insert=0.0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=collate_fn, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers
        )


class DocumentSeparationModuleJSON(pl.LightningDataModule):
    def __init__(
        self,
        json_files_train: Sequence[Path],
        json_files_val: Optional[Sequence[Path]],
        transform=None,
        batch_size: int = 8,
        number_of_images: int = 3,
        num_workers: int = 8,
        sample_same_inventory: bool = False,
        wrap_round: bool = False,
        prob_shuffle_document: float = 0.0,
        prob_randomize_document_order: float = 0.0,
        prob_random_scan_insert: float = 0.0,
        split_ratio: float = 0.8,
    ):
        super().__init__()

        if json_files_val is None:
            # split training paths into training and validation paths
            training_paths, val_paths = split_training_paths(json_files_train, split_ratio=split_ratio, seed=101)
        else:
            val_paths = natsorted(json_files_val)

        self.transform = transform
        self.batch_size = batch_size
        self.number_of_images = number_of_images
        self.num_workers = num_workers

        self.sample_same_inventory = sample_same_inventory
        self.wrap_round = wrap_round

        self.prob_shuffle_document = prob_shuffle_document
        self.prob_randomize_document_order = prob_randomize_document_order
        self.prob_random_scan_insert = prob_random_scan_insert

        self.training_paths = self.jsons_to_paths(training_paths)
        self.val_paths = self.jsons_to_paths(val_paths)

    @staticmethod
    def scan_id_to_inventory_number(scan_id: str) -> str:
        if check := re.match(r"(.+)_(.+)_(\d+)(_deelopname\d+)?", scan_id):
            inventory_number_file = check.group(2)
            return inventory_number_file
        else:
            raise ValueError(f"Scan id {scan_id} does not match the expected format")

    @staticmethod
    def json_to_scan_label(path: Path) -> tuple[str, Path, bool]:
        with path.open(mode="r") as f:
            data = json.load(f)
            is_first_page = data["isFirstPage"]
            scan_id = data["scanId"]

            # HACK This is a hack to get the correct file name
            base = Path("/data/spinque-converted/")

            inventory_number = DocumentSeparationModuleJSON.scan_id_to_inventory_number(scan_id)
            inventory_number_dir = path.parent.name
            if inventory_number_dir != inventory_number:
                logger.warning(
                    f"Inventory number in dir {inventory_number_dir} does not match with inventory number in file {inventory_number}. Path: {path}"
                )
            file_name = base.joinpath(inventory_number, f"{scan_id}.jp2")

        return inventory_number, file_name, is_first_page

    def jsons_to_paths(self, json_paths: Sequence[Path]) -> Sequence[Sequence[Sequence[Path]]]:
        paths = []
        current_inventory_number = None
        current_document = []
        current_inventory = []
        for json_path in natsorted(json_paths):
            inventory_number, file_name, is_first_page = self.json_to_scan_label(json_path)

            if current_inventory_number is None:
                current_inventory_number = inventory_number
                current_document = [file_name]
                continue
            if current_inventory_number != inventory_number:
                current_inventory.append(current_document)
                paths.append(current_inventory)

                current_inventory_number = inventory_number
                current_inventory = []
                current_document = [file_name]
            else:
                if is_first_page:
                    current_inventory.append(current_document)
                    current_document = [file_name]
                else:
                    current_document.append(file_name)

        if current_document:
            current_inventory.append(current_document)
            paths.append(current_inventory)

        return paths

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DocumentSeparationDataset(
                self.training_paths,
                number_of_images=self.number_of_images,
                transform=self.transform,
                sample_same_inventory=self.sample_same_inventory,
                wrap_round=self.wrap_round,
                prob_shuffle_document=self.prob_shuffle_document,
                prob_randomize_document_order=self.prob_randomize_document_order,
                prob_random_scan_insert=self.prob_random_scan_insert,
            )
            self.val_dataset = DocumentSeparationDataset(
                self.val_paths,
                number_of_images=self.number_of_images,
                transform=self.transform,
                sample_same_inventory=True,
                wrap_round=False,
                prob_shuffle_document=0.0,
                prob_randomize_document_order=0.0,
                prob_random_scan_insert=0.0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=collate_fn, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers
        )
