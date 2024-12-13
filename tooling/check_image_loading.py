import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.input_utils import get_file_paths, supported_image_formats


def get_arguments():
    parser = argparse.ArgumentParser(description="Check if image files will load")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    """
    Check if image files will load

    Args:
        args: Arguments from the command line
    """
    stats = defaultdict(lambda: defaultdict(int))

    # load all image files
    file_paths = get_file_paths(args.input, formats=supported_image_formats)

    for file_path in tqdm(file_paths):
        extension = file_path.suffix.lower()
        stats[extension]["total"] += 1

        # Check if the image will load
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            image = Image.open(file_path)
            image.load()
            stats[extension]["success"] += 1
            if np.all(np.asarray(image) == 0):
                stats[extension]["blank"] += 1
        except OSError:
            # Try to load the image with truncated images
            stats[extension]["failure"] += 1

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                image = Image.open(file_path)
                image.load()
                stats[extension]["success_truncated"] += 1
                if np.all(np.asarray(image) == 0):
                    stats[extension]["blank_truncated"] += 1
            except OSError:
                stats[extension]["failure_truncated"] += 1

    # Print the stats
    for extension in stats:
        print(f"Extension: {extension}")
        for key in stats[extension]:
            print(f"{key}: {stats[extension][key]}")
        print("")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
