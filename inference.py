import argparse
import itertools
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from natsort import natsorted
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

from data.augmentations import PadToMaxSize, SmartCompose

# from data.dataloader import collate_fn
from data.dataset import DocumentSeparationDataset
from models.model14 import DocumentSeparator, ImageEncoder, TextEncoder
from utils.input_utils import get_file_paths, supported_image_formats

torch.set_float32_matmul_precision("high")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file for Document Separation")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/files", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)
    io_args.add_argument("-c", "--checkpoint", help="Path to the checkpoint", type=str, default=None)
    io_args.add_argument(
        "--thumbnail_dir",
        help="Path to the thumbnail directory",
        type=str,
        default="/data/thumbnails/data/spinque-converted",
    )

    io_args.add_argument("--override", help="Override existing results", action="store_true")

    args = parser.parse_args()
    return args


class Predictor:
    def __init__(
        self,
        checkpoint: str | Path,
    ) -> None:
        if not checkpoint:
            raise ValueError("No checkpoint provided")

        self.model = DocumentSeparator.load_from_checkpoint(checkpoint, offline=True)
        self.model.eval()

    def __call__(self, data: dict) -> dict:
        with torch.no_grad():
            result_logits, _, _ = self.model(data)
        # result_logits = torch.nn.functional.softmax(result_logits, dim=1)
        result_logits = torch.nn.functional.sigmoid(result_logits)
        confidence = torch.where(result_logits > 0.5, result_logits, 1 - result_logits)
        label = (result_logits > 0.5).to(torch.int64)
        output = {
            "label": label,
            "confidence": confidence,
        }
        return output


def collate_fn(batch):
    _images = [item["images"] for item in batch]
    shapes = [item["shapes"] for item in batch]
    texts = [item["texts"] for item in batch]
    image_paths = [item["image_paths"] for item in batch]

    # Pad to the same size

    if all(image is None for sub_images in _images for image in sub_images):
        # keep shape
        batch_size = len(batch)
        images_size = len(_images[0])
        images = torch.zeros((batch_size, images_size, 3, 1, 1))
    else:
        max_shape = np.max([image.size()[-2:] for sub_images in _images for image in sub_images if image is not None], axis=0)
        images = []
        for i in range(len(_images)):
            for j in range(len(_images[i])):
                if _images[i][j] is None:
                    _images[i][j] = torch.zeros((3, max_shape[0], max_shape[1]))
                _images[i][j] = torch.nn.functional.pad(
                    _images[i][j],
                    (0, int(max_shape[1] - _images[i][j].size()[-1]), 0, int(max_shape[0] - _images[i][j].size()[-2])),
                    value=0,
                )
            images.append(torch.stack(_images[i]))

        images = torch.stack(images)
    shapes = torch.tensor(shapes)

    return {
        "images": images,
        "shapes": shapes,
        "texts": texts,
        "image_paths": image_paths,
    }


class SavePredictor(Predictor):
    def __init__(
        self,
        input_paths: list[Path],
        output_path: Path,
        checkpoint: str,
        override: bool = False,
        thumbnail_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(checkpoint)
        self.input_paths = None
        self.output_dir = None
        self.override = override
        if output_path is not None:
            self.set_output_path(output_path)

        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.transform = SmartCompose(
            [
                ToTensor(),
                PadToMaxSize(),
                Resize((512, 512)),
            ]
        )

        self.thumbnail_dir = thumbnail_dir

    def results_exist(self, path: Path) -> bool:
        if self.output_dir is None:
            raise ValueError("Output path not set")
        for image_path in get_file_paths(path, formats=supported_image_formats):
            json_path = self.output_dir.joinpath(image_path.parent.name, image_path.stem + ".json")
            if not json_path.exists():
                return False
        print(f"Results exist {self.output_dir.joinpath(path.name)}. Skipping inventory {path.name}")
        return True

    def set_input_paths(self, input_paths: list[Path]) -> None:
        input_paths = natsorted([Path(input_path) for input_path in input_paths])
        all_input_paths = []
        for path in input_paths:
            if path.is_file() and path.suffix == ".txt":
                with open(path, "r") as f:
                    _input_paths = [Path(line.strip()) for line in f.readlines()]
                for _input_path in _input_paths:
                    if not _input_path.is_dir():
                        raise FileNotFoundError(f"Could not find {_input_path}")
                all_input_paths.extend(natsorted(_input_paths))
            elif path.is_dir():
                all_input_paths.append(path)
            else:
                raise ValueError(f"Input path {path} is not a file or directory")
        assert len(set(input_path.name for input_path in all_input_paths)) == len(
            all_input_paths
        ), "All input paths must have a unique name"

        grouped_paths = [
            [get_file_paths(input_path, formats=supported_image_formats)]
            for input_path in tqdm(all_input_paths, desc="Getting file paths for each inventory")
            if self.override or not self.results_exist(input_path)
        ]

        self.input_paths = grouped_paths

    def set_output_path(self, output_dir: Path | str) -> None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            print(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)
        self.output_dir = output_dir

    def get_middle_scan(self, y):
        return y[:, y.shape[1] // 2]

    def get_middle_path(self, paths: list[list[Path]]):
        return paths[0][len(paths[0]) // 2]

    def process(self):
        if self.input_paths is None:
            raise TypeError("No input paths provided")
        if self.output_dir is None:
            raise ValueError("Output path not set")

        dataset = DocumentSeparationDataset(
            image_paths=self.input_paths,
            mode="test",
            transform=self.transform,
            number_of_images=3,
            wrap_round=False,
            sample_same_inventory=True,
            prob_random_scan_insert=0.0,
            prob_randomize_document_order=0.0,
            prob_shuffle_document=0.0,
            thumbnail_dir=self.thumbnail_dir,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=16,
            collate_fn=collate_fn,
        )

        inventory = None
        for data in tqdm(dataloader, desc="Processing"):
            result = self(data)

            middle_path = self.get_middle_path(data["image_paths"])
            inventory = middle_path.parent

            output_path = self.output_dir.joinpath(inventory.name, middle_path.stem + ".json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result = {
                "result": result["label"].cpu().item(),
                "confidence": result["confidence"].cpu().item(),
            }

            with output_path.open("w") as f:
                json.dump(result, f)


def main(args):
    predictor = SavePredictor(args.input, args.output, args.checkpoint, args.override, args.thumbnail_dir)
    predictor.process()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
