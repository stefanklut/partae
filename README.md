# document-separation

This is a tool to find what scans belong to the same document. It uses AI to compare the images and the text of consecutive scans and determine if they belong to the same document. It classifies a scan as either the start of a document or a continuation of the previous document. 

## Table of Contents
- [document-separation](#document-separation)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Conda](#conda)
    - [Docker](#docker)
      - [Manual Installation](#manual-installation)
  - [Training](#training)
    - [Examples](#examples)


## Setup
The recommended way of running this tool is inside a conda environment. To ensure easier compatibility a method of building a docker is also provided.

To start clone the github repo to your local machine using either HTTPS:
```sh
git clone https://github.com/stefanklut/partae.git
```

Or using SSH:
```sh
git clone git@github.com:stefanklut/partae.git
```


And make go to the working directory:
```sh
cd partae
```

### Conda
If not already installed, install either conda or miniconda ([install instructions][conda_install_link]), or mamba ([install instructions][mamba_install_link]). 

The required packages are listed in the [`environment.yml`][environment_link] file. The environment can be automatically created using the following commands.

Using conda/miniconda:
```sh
conda env create -f environment.yml
```

Using mamba:
```sh
mamba env create -f environment.yml
```

When running the tool always activate the conda environment
```sh
conda activate partae
```

### Docker
If not already installed, install the Docker Engine ([install instructions][docker_install_link]). The docker environment can most easily be build with the provided script.

#### Manual Installation
Building the docker using the provided script:
```sh
./buildImage.sh <PATH_TO_DIR>
```

Or the multistage build with some profiler tools taken out (might be smaller):
```sh
./buildImage.multistage.sh <PATH_TO_DIR>
```

<details>
<summary> Click for manual docker install instructions (not recommended) </summary>

First copy the Laypa directory to the temporary docker directory:
```sh
tmp_dir=$(mktemp -d)
cp -r -T <PATH_TO_DIR> $tmp_dir/separation
cp Dockerfile $tmp_dir/Dockerfile
cp _entrypoint.sh $tmp_dir/_entrypoint.sh
cp .dockerignore $tmp_dir/.dockerignore
```

Then build the docker image using the following command:
```sh
docker build -t docker.separation $tmp_dir
```
</details>

<!-- TODO Update the training and inference section -->

## Training

The model is trained using the `main.py` script. The model reads a xlsx file (`-x/--xlsx`) with the following columns: "Start of document" or "URL nieuw document op volgorde". It then reads the images responding to the URLs in the column matching inventory number. This is done using the `-t/--train` parameters which requires the path to the images or a directory containing the images. See the `-h/--help` for more information.


### Examples
If validation is not specified the default is 0.2. The validation is the percentage of the data that is used for validation. The rest is used for training.
```sh
python main.py -x <path_to_xlsx> -t <path_to_images>
```

To specify the validation folders:
```sh
python main.py -x <path_to_xlsx> -t <path_to_images> -v <path_to_validation>
```
