import argparse
import subprocess
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import Resize, ToTensor

from core.trainer import ClassificationModel
from data.augmentations import PadToMaxSize, SmartCompose
from data.datamodule import DocumentSeparationModule
from models.model import DocumentSeparator, ImageEncoder, TextEncoder
from utils.input_utils import get_file_paths, supported_image_formats

torch.set_float32_matmul_precision("high")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Document Separation")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-t", "--train", help="Train input folder/file", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument(
        "-v", "--val", help="Validation input folder/file", nargs="+", action="extend", type=str, required=False
    )
    io_args.add_argument("-x", "--xlsx", help="XLSX file with labels", type=str, default=None)

    args = parser.parse_args()
    return args


def get_git_hash() -> str:
    version_path = Path("version_info")

    if version_path.is_file():
        with version_path.open(mode="r") as file:
            git_hash = file.read().strip()
    else:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent).strip().decode()
    return git_hash


def main(args: argparse.Namespace):
    training_paths = get_file_paths(args.train, formats=supported_image_formats)
    if args.val is None:
        val_paths = None
    else:
        val_paths = get_file_paths(args.val, formats=supported_image_formats)
    xlsx_file = Path(args.xlsx)
    if not xlsx_file.exists():
        raise FileNotFoundError(f"XLSX file {xlsx_file} does not exist")

    transform = SmartCompose(
        [
            ToTensor(),
            PadToMaxSize(),
            Resize((224, 224)),
        ]
    )

    data_module = DocumentSeparationModule(
        training_paths=training_paths,
        val_paths=val_paths,
        xlsx_file=xlsx_file,
        transform=transform,
        batch_size=4,
        number_of_images=3,
        num_workers=8,
    )
    model = ClassificationModel(
        model=DocumentSeparator(
            image_encoder=ImageEncoder(merge_to_batch=True),
            text_encoder=TextEncoder(merge_to_batch=True),
        ),
        learning_rate=1e-5,
    )

    logger = TensorBoardLogger("lightning_logs", name="document_separator")
    output_dir = Path(logger.log_dir)

    # Save git hash to output directory
    git_hash = get_git_hash()
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("git_hash").open("w") as file:
        file.write(git_hash)

    checkpointer = ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir.joinpath("checkpoints"),
        filename="document_separator-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=3,
        callbacks=[checkpointer],
        val_check_interval=0.25,
        logger=logger,
        detect_anomaly=True,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
