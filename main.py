import argparse

import lightning as pl

from core.trainer import ClassificationModel
from models.model import DocumentSeparator, ImageEncoder, TextEncoder


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Document Separation")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-t", "--train", help="Train input folder/file", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument(
        "-v", "--val", help="Validation input folder/file", nargs="+", action="extend", type=str, required=True
    )

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    model = ClassificationModel(
        model=DocumentSeparator(image_encoder=ImageEncoder(merge_to_batch=True), text_encoder=TextEncoder(merge_to_batch=True))
    )
    trainer = pl.Trainer(max_epochs=10)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
