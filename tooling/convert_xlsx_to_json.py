import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from data.convert_xlsx import link_with_paths
from utils.input_utils import get_file_paths, supported_image_formats


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert xlsx to json")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-x", "--xlsx", help="XLSX file with labels", type=str, required=True)
    io_args.add_argument("-i", "--input", help="Train input folder/file", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)

    args = parser.parse_args()

    return args


def main(args):
    paths = get_file_paths(args.input, supported_image_formats)
    all_documents = link_with_paths(args.xlsx, paths)

    output_path = Path(args.output)

    for inventory in all_documents:
        for document in inventory:
            for i, image in enumerate(document):
                inventory = Path(image).parent.name
                output_path_image = output_path.joinpath(inventory, f"{image.stem}.json")

                output_path_image.parent.mkdir(parents=True, exist_ok=True)
                if i == 0:
                    is_first_page = True
                else:
                    is_first_page = False
                content = {"scanId": image.stem, "isFirstPage": is_first_page, "user": "converted-from-xlsx"}
                with output_path_image.open("w") as f:
                    json.dump(content, f)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
