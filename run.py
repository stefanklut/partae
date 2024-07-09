import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from natsort import natsorted
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

from core.trainer import ClassificationModel
from data.augmentations import PadToMaxSize, SmartCompose

# from data.dataloader import collate_fn
from data.dataset import DocumentSeparationDataset
from models.model import DocumentSeparator, ImageEncoder, TextEncoder
from utils.input_utils import get_file_paths, supported_image_formats


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
        model = DocumentSeparator(
            image_encoder=ImageEncoder(merge_to_batch=True),
            text_encoder=TextEncoder(merge_to_batch=True),
            output_size=2,
        )

        if not checkpoint:
            raise ValueError("No checkpoint provided")

        self.model = ClassificationModel.load_from_checkpoint(checkpoint, model=model)
        self.model = model

    def __call__(self, data: dict) -> torch.Tensor:
        result_logits = self.model(data)
        result = torch.argmax(result_logits, dim=2)
        return result


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

        training_paths = natsorted(paths)
        # Group the paths by parent folder
        grouped_paths = [list(group) for _, group in itertools.groupby(training_paths, key=lambda x: x.parent)]
        paths = [[group] for group in grouped_paths]

        self.input_paths = paths

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

        for data in tqdm(dataloader, desc="Processing"):
            result = self(data)

            result = result.cpu().numpy()
            middle_scan = self.get_middle_scan(result)
            middle_path = self.get_middle_path(data["image_paths"])
            # Save the result
            name = middle_path.stem

            output_path = self.output_dir / f"{name}.txt"
            with open(output_path, "w") as file:
                file.write(str(middle_scan.item()))


def main(args):
    predictor = SavePredictor(args.input, args.output, args.checkpoint)
    predictor.process()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
