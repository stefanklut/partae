import argparse
import sys
from pathlib import Path

import torch
import torch.utils.data
from torchmetrics import ConfusionMatrix
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from data.augmentations import PadToMaxSize, SmartCompose
from data.convert_xlsx import link_with_paths
from data.dataloader import collate_fn
from data.dataset import DocumentSeparationDataset
from models.model1 import DocumentSeparator, ImageEncoder, TextEncoder
from models.rules_based import RulesBased
from utils.input_utils import get_file_paths, supported_image_formats

torch.set_float32_matmul_precision("high")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Document Separation")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-v", "--val", help="Validation input folder/file", nargs="+", action="extend", type=str, required=False
    )
    io_args.add_argument("-x", "--xlsx", help="XLSX file with labels", type=str, default=None)
    io_args.add_argument("-c", "--checkpoint", help="Path to the checkpoint", type=str, default=None)
    io_args.add_argument(
        "--thumbnail_dir",
        help="Path to the thumbnail directory",
        type=str,
        default=Path("/data/thumbnails/data/spinque-converted"),
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    """
    Evaluate the model, based on the validation set given in xlsx file
    """

    # TODO Update to also use the json format of GT
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

    val_paths = link_with_paths(xlsx_file, val_paths)

    dataset = DocumentSeparationDataset(
        image_paths=val_paths,
        transform=transform,
        number_of_images=3,
        thumbnail_dir=args.thumbnail_dir,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_fn,
    )

    def get_middle_scan(y):
        return y[:, y.shape[1] // 2]

    def split_input(batch):
        y = batch["targets"]
        del batch["targets"]
        return batch, y

    if args.checkpoint:
        model = DocumentSeparator.load_from_checkpoint(args.checkpoint)

    rules = RulesBased()

    # Evaluate the model
    model.eval()
    confusion_matrix_model = ConfusionMatrix(task="multiclass", num_classes=2).to(model.device)
    confusion_matrix_rules = ConfusionMatrix(task="multiclass", num_classes=2).to(model.device)
    for batch in tqdm(val_dataloader):
        x, y = split_input(batch)
        y = get_middle_scan(y)

        y_hat_model, _, _ = model(x)
        y_hat_model = torch.argmax(y_hat_model, dim=1)
        confusion_matrix_model.update(y_hat_model, y.to(model.device))

        y_hat_rules = rules(x)
        y_hat_rules = torch.argmax(y_hat_rules, dim=1).to(model.device)
        confusion_matrix_rules.update(y_hat_rules, y.to(model.device))

    if args.checkpoint:
        print(confusion_matrix_model.compute())

    print(confusion_matrix_rules.compute())


if __name__ == "__main__":
    args = get_arguments()
    main(args)
