import argparse
from pathlib import Path

from natsort import natsorted


def get_arguments():
    parser = argparse.ArgumentParser(description="Split data")
    parser.add_argument("-i", "--input", help="Input folder", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output folder", type=str)

    args = parser.parse_args()
    return args


def main(args):
    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as f_in:
        paths = f_in.readlines()
    with output_path.open("w") as f_out:
        for path in natsorted(paths):
            f_out.write(path)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
