import argparse
import sys
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.input_utils import get_file_paths, supported_image_formats


def get_arguments():
    parser = argparse.ArgumentParser(description="Check if image files will load")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, default="log_image_loading")
    args = parser.parse_args()
    return args


def collate_fn(batch):
    paths = [data["path"] for data in batch]
    success_thumbnail = [data["success_thumbnail"] for data in batch]
    success_image = [data["success_image"] for data in batch]
    return {"path": paths, "success_thumbnail": success_thumbnail, "success_image": success_image}


class LogDataset(Dataset):
    """
    Dataset that logs the success of loading an image
    """

    def __init__(self, image_paths):
        super(Dataset, self).__init__()
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, image_path: Path) -> dict:
        """
        Load an image and return the success of loading the image

        Args:
            image_path (Path): Path to the image

        Returns:
            dict: Dictionary containing the path to the image and the success of loading the thumbnail and the image
        """
        # Check if thumbnail exists

        failed_thumbnail = False
        failed_image = False

        thumbnail_path = Path("/data/thumbnails/").joinpath(str(image_path.relative_to(Path("/"))) + ".thumbnail.jpg")
        try:
            image = Image.open(thumbnail_path)
            image.load()
            image = image.convert("RGB")
        except OSError as e:
            failed_thumbnail = True
            print(f"Could not open thumbnail {thumbnail_path}. Trying to open original image")
            try:
                image = Image.open(image_path.resolve())
                image.load()
                image = image.convert("RGB")
            except OSError as e:
                failed_image = True
                print(f"Could not open image {image_path}")
        return {"path": image_path, "success_thumbnail": not failed_thumbnail, "success_image": not failed_image}

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        return self.load_image(image_path)


def main(args):
    image_paths = get_file_paths(args.input, formats=supported_image_formats)
    dataset = LogDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=collate_fn)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path.joinpath("thumbnails"), "w") as f_thumbnail, open(output_path.joinpath("images"), "w") as f_image:
        for data in tqdm(dataloader):
            path = data["path"][0]
            success_thumbnail = data["success_thumbnail"][0]
            success_image = data["success_image"][0]
            if not success_thumbnail:
                f_thumbnail.write(f"{path}\n")
            if not success_image:
                f_image.write(f"{path}\n")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
