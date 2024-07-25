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

    io_args.add_argument("-c", "--checkpoint", help="Checkpoint file", type=str, default=None)

    training_args = parser.add_argument_group("Training")
    training_args.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=10)
    training_args.add_argument("-b", "--batch_size", help="Batch size", type=int, default=32)
    training_args.add_argument("--num_workers", help="Number of workers", type=int, default=16)
    training_args.add_argument("--learning_rate", help="Learning rate", type=float, default=1e-5)
    training_args.add_argument("--optimizer", help="Optimizer", type=str, default="Adam")
    training_args.add_argument("--label_smoothing", help="Label smoothing", type=float, default=0.0)

    dataset_args = parser.add_argument_group("Dataset")
    dataset_args.add_argument("-n", "--number_of_images", help="Number of images", type=int, default=3)
    dataset_args.add_argument("--randomize_document_order", help="Randomize document order", action="store_true")
    dataset_args.add_argument("--sample_same_inventory", help="Sample same inventory", action="store_true")
    dataset_args.add_argument("--wrap_round", help="Wrap round", action="store_true")

    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--turn_off_image", help="Turn off image encoder", action="store_true")
    model_args.add_argument("--turn_off_text", help="Turn off text encoder", action="store_true")
    model_args.add_argument("--turn_off_shapes", help="Turn off shapes encoder", action="store_true")
    model_args.add_argument("--dropout", help="Dropout", type=float, default=0.5)

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

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = None

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
        batch_size=args.batch_size,
        number_of_images=args.number_of_images,
        num_workers=args.num_workers,
        randomize_document_order=args.randomize_document_order,
        sample_same_inventory=args.sample_same_inventory,
        wrap_round=args.wrap_round,
    )
    model = ClassificationModel(
        model=DocumentSeparator(
            image_encoder=ImageEncoder(merge_to_batch=True),
            text_encoder=TextEncoder(merge_to_batch=True),
            turn_off_image=args.turn_off_image,
            turn_off_text=args.turn_off_text,
            turn_off_shapes=args.turn_off_shapes,
            dropout=args.dropout,
        ),
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        label_smoothing=args.label_smoothing,
    )

    logger = TensorBoardLogger("lightning_logs", name="document_separator")
    output_dir = Path(logger.log_dir)

    # Save git hash to output directory
    git_hash = get_git_hash()
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("git_hash").open("w") as file:
        file.write(git_hash)

    # Save arguments to output directory
    with output_dir.joinpath("arguments").open("w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    checkpointer_val_loss = ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir.joinpath("checkpoints"),
        filename="document_separator-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        save_last="link",
        mode="min",
    )

    checkpointer_val_center_acc = ModelCheckpoint(
        monitor="val_center_acc",
        dirpath=output_dir.joinpath("checkpoints"),
        filename="document_separator-{epoch:02d}-{val_center_acc:.4f}",
        save_top_k=3,
        mode="max",
    )

    checkpointer_epoch = ModelCheckpoint(
        dirpath=output_dir.joinpath("checkpoints"),
        filename="document_separator-{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last="link",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpointer_val_loss, checkpointer_epoch, checkpointer_val_center_acc],
        val_check_interval=0.25,
        logger=logger,
    )

    trainer.fit(model, data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
