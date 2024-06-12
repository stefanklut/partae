import argparse
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

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

    args = parser.parse_args()
    return args


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
        )
    )

    logger = TensorBoardLogger("lightning_logs", name="document_separator")
    output_dir = Path(logger.log_dir).joinpath("checkpoints")
    print(output_dir)

    checkpointer = ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir,
        filename="document_separator-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpointer],
        val_check_interval=0.5,
        logger=logger,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
