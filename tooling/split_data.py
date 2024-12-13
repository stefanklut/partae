import argparse
from pathlib import Path

from natsort import natsorted


def get_arguments():
    parser = argparse.ArgumentParser(description="Split data")
    parser.add_argument("-i", "--input", help="Input folder", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output folder", type=str, required=True)

    args = parser.parse_args()
    return args


def main(args):
    """
    Method from splitting data into a file. The file will have a list of directories from the input folder.

    Args:
        args: Arguments from the command line
    """
    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.mkdir(parents=True, exist_ok=True)

    with output_path.joinpath("split.txt").open("w") as f:
        for path in natsorted(list(input_path.glob("*/"))):
            path = input_path.joinpath(path)
            f.write(str(path) + "\n")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
