import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from natsort import natsorted
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

from data.augmentations import PadToMaxSize, SmartCompose

# from data.dataloader import collate_fn
from data.dataset import DocumentSeparationDataset
from models.model1 import DocumentSeparator, ImageEncoder, TextEncoder
from utils.input_utils import get_file_paths, supported_image_formats

torch.set_float32_matmul_precision("high")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file for Document Separation")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/files", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)
    io_args.add_argument("-c", "--checkpoint", help="Path to the checkpoint", type=str, default=None)

    args = parser.parse_args()
    return args


class Predictor:
    def __init__(
        self,
        checkpoint: str | Path,
    ) -> None:
        if not checkpoint:
            raise ValueError("No checkpoint provided")

        self.model = DocumentSeparator.load_from_checkpoint(checkpoint)
        self.model.eval()

    def __call__(self, data: dict) -> dict:
        result_logits, _, _ = self.model(data)
        result_logits = torch.nn.functional.softmax(result_logits, dim=1)
        confidence, label = torch.max(result_logits, dim=1)
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
    ) -> None:
        super().__init__(checkpoint)
        self.input_paths = None
        if input_paths is not None:
            self.set_input_paths(input_paths)

        if output_path is not None:
            self.set_output_path(output_path)

        self.transform = SmartCompose(
            [
                ToTensor(),
                PadToMaxSize(),
                Resize((224, 224)),
            ]
        )

    def set_input_paths(self, input_paths: list[Path]) -> None:
        paths = get_file_paths(input_paths, formats=supported_image_formats)

        input_paths = natsorted(paths)
        # Group the paths by parent folder
        grouped_paths = [list(group) for _, group in itertools.groupby(input_paths, key=lambda x: x.parent)]
        grouped_paths = [[group] for group in grouped_paths]

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

        dataset = DocumentSeparationDataset(
            image_paths=self.input_paths,
            mode="test",
            transform=None,
            number_of_images=3,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=collate_fn,
        )

        def save_json(json_data, inventory_name):
            json_data = {k: v for d in json_data for k, v in d.items()}
            output_path = self.output_dir.joinpath(inventory_name, "results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving results to {output_path}")
            with open(output_path, "w") as f:
                json.dump(json_data, f)

        json_data = []
        previous_inventory = None
        for data in tqdm(dataloader, desc="Processing"):
            result = self(data)

            middle_path = self.get_middle_path(data["image_paths"])
            inventory = middle_path.parent
            if inventory != previous_inventory:
                if previous_inventory is not None:
                    inventory_name = previous_inventory.name
                    save_json(json_data, inventory_name)
                previous_inventory = inventory
                json_data = []
            # Save the result

            result = {
                str(middle_path): {
                    "result": result["label"].cpu().item(),
                    "confidence": result["confidence"].cpu().item(),
                }
            }
            json_data.append(result)

        save_json(json_data, inventory)


def main(args):
    predictor = SavePredictor(args.input, args.output, args.checkpoint)
    predictor.process()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
