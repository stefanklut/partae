import argparse
import functools
import json
import logging
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted, os_sorted
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization of prediction/GT of model")

    io_args = parser.add_argument_group("IO")
    # io_args.add_argument("-t", "--train", help="Train input folder/file",
    #                         nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-o", "--output", help="Output folder", type=str)

    args = parser.parse_args()

    return args


IMAGE_PRELOAD = 16

_keypress_result = None


def keypress(event):
    global _keypress_result
    # print('press', event.key)
    if event.key in ["q", "escape"]:
        sys.exit()
    if event.key in ["right"]:
        _keypress_result = "forward"
        return
    if event.key in [" "]:
        _keypress_result = "flip"
        return
    if event.key in ["left"]:
        _keypress_result = "back"
        return
    if event.key in ["e", "delete"]:
        _keypress_result = "delete"
        return
    if event.key in ["w"]:
        _keypress_result = "bad"
        return


def on_close(event):
    sys.exit()


@functools.lru_cache(maxsize=IMAGE_PRELOAD * 2)
def get_image(image_path: str) -> Optional[np.ndarray]:
    image_path = Path(image_path)

    # Check if thumbnail exists
    thumbnail_path = Path("/data/thumbnails/").joinpath(str(image_path.relative_to(Path("/"))) + ".thumbnail.jpg")
    try:
        image = Image.open(thumbnail_path)
        image.load()
        image = image.convert("RGB")
    except OSError as e:
        print(f"Could not open thumbnail {thumbnail_path}. Trying to open original image")
        try:
            image = Image.open(image_path.resolve())
            image.load()
            image = image.convert("RGB")
        except OSError as e:
            print(f"Could not open image {image_path}")
            return None
    return np.asarray(image)


def scan_id_to_inventory_number(scan_id: str) -> str:
    if check := re.match(r"(.+)_(.+)_(\d+)(_deelopname\d+)?", scan_id):
        inventory_number_file = check.group(2)
        return inventory_number_file
    else:
        raise ValueError(f"Scan id {scan_id} does not match the expected format")


def json_to_scan_label(path: Path) -> tuple[Path, bool]:
    with path.open(mode="r") as f:
        data = json.load(f)
        is_first_page = data["isFirstPage"]
        scan_id = data["scanId"]
        # HACK This is a hack to get the correct file name
        base = Path("/data/spinque-converted/")
        inventory_number = scan_id_to_inventory_number(scan_id)
        inventory_number_dir = path.parent.name
        if inventory_number_dir != inventory_number:
            logger.warning(
                f"Inventory number in dir {inventory_number_dir} does not match with inventory number in file {inventory_number}. Path: {path}"
            )
        file_name = base.joinpath(inventory_number, f"{scan_id}.jp2")
    return file_name, is_first_page


def invert_json_is_first_page(path: Path) -> None:
    with path.open(mode="r") as f:
        data = json.load(f)
        data["isFirstPage"] = not data["isFirstPage"]
        data["user"] = "stefank"
    with path.open(mode="w") as f:
        json.dump(data, f)


def main(args) -> None:
    """
    Currently running the validation set and showing the ground truth and the prediction side by side

    Args:
        args (argparse.Namespace): arguments for where to find the images
    """
    json_paths = []
    for json_path in args.input:
        json_path = Path(json_path)
        if json_path.is_dir():
            for path in json_path.rglob("*.json"):
                # file_name, is_first_page = json_to_scan_label(path)
                # combined_jsons[file_name] = {"result": is_first_page}
                json_paths.append(path)
        if json_path.is_file():
            assert json_path.suffix == ".json", f"File {json_path} is not a json file"
            # file_name, is_first_page = json_to_scan_label(json_path)
            # combined_jsons[file_name] = {"result": is_first_page}
            json_paths.append(json_path)

    loader = os_sorted(json_paths)

    fig, axes = plt.subplots(1)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]  # Ensure axes is a list
    fig.tight_layout()
    fig.canvas.mpl_connect("key_press_event", keypress)
    fig.canvas.mpl_connect("close_event", on_close)
    axes[0].axis("off")
    fig_manager = plt.get_current_fig_manager()
    if fig_manager is None:
        raise ValueError("Could not find figure manager")
    fig_manager.window.showMaximized()

    pool = ThreadPool(4)

    for i in range(min(IMAGE_PRELOAD, len(loader))):
        preload_image_path, _ = json_to_scan_label(loader[i])
        pool.submit(get_image, preload_image_path)

    i = 0
    while 0 <= i < len(loader):
        json_path = loader[i]

        image_path, is_first_page = json_to_scan_label(json_path)

        fig_manager.window.setWindowTitle(str(image_path))

        # HACK Just remove the previous axes, I can't find how to resize the image otherwise
        axes[0].clear()
        axes[0].axis("off")

        image = get_image(image_path)

        border = 10
        color = [255, 0, 0] if is_first_page == 1 else [255, 255, 255]
        image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=color)

        if image is None:
            image = np.zeros((100, 100, 3), dtype=np.uint8)

        axes[0].imshow(image)
        if i + IMAGE_PRELOAD < len(loader):
            preload_image_path, _ = json_to_scan_label(loader[i + IMAGE_PRELOAD])
            pool.submit(get_image, preload_image_path)

        suptitle = f"{i+1}/{len(loader)}: {Path(image_path).name} result: {is_first_page}"
        fig.suptitle(suptitle)

        # f.title(inputs["file_name"])

        global _keypress_result
        _keypress_result = None
        fig.canvas.draw()
        while _keypress_result is None:
            plt.waitforbuttonpress()
        if _keypress_result == "forward":
            # print(i+1, f"{inputs['original_file_name']}")
            i += 1
        elif _keypress_result == "back":
            # print(i+1, f"{inputs['original_file_name']}: DELETE")
            i -= 1
        elif _keypress_result == "flip":
            invert_json_is_first_page(json_path)

    pool.shutdown()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
