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

The model is trained using the `train.py` script. Generally the script requires the paths to json files or the directory in which these json files are stored. 

```sh
python train.py --train <path_to_json> --val <path_to_json>
```

If you do not specify the `--val` argument the script will use a 80/20 split of the training data for validation.
```sh
python train.py --train <path_to_json>
```

To change the augmentation parameters use the following arguments:
```sh
  -n/--number_of_images NUMBER_OF_IMAGES
                        Number of images
  --prob_shuffle_document PROB_SHUFFLE_DOCUMENT
                        Probability to shuffle document
  --prob_randomize_document_order PROB_RANDOMIZE_DOCUMENT_ORDER
                        Probability to randomize document order
  --prob_random_scan_insert PROB_RANDOM_SCAN_INSERT
                        Probability to insert random scan
  --sample_all_inventories
                        Sample all inventories
  --wrap_round          
                        Wrap round
  --split_ratio SPLIT_RATIO
                        Split ratio
```

To change factors for the training use the following arguments:
```sh
  -e/--epochs EPOCHS
                        Number of epochs
  -b/--batch_size BATCH_SIZE
                        Batch size
  --num_workers NUM_WORKERS
                        Number of workers
  --learning_rate LEARNING_RATE
                        Learning rate
  --optimizer OPTIMIZER
                        Optimizer
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing
  --unfreeze_imagenet UNFREEZE_IMAGENET
                        Unfreeze ImageNet after epochs or percentage of epochs
  --unfreeze_roberta UNFREEZE_ROBERTA
                        Unfreeze RoBERTa after epochs or percentage of epochs
  --dropout DROPOUT     
                        Dropout
```

To continue training from a checkpoint use the following argument:
```sh
  --checkpoint CHECKPOINT
                        Checkpoint
```

To change the name of the run and the output directory use the following arguments:
```sh
  --name NAME           
                        Name of the run
  --o/output OUTPUT
                        Output folder
```

To use the XLSX format instead of the JSON format use the following arguments and change the `--train` and `--val` arguments to the paths of the images:
```sh
  --use_xlsx            
                        Use XLSX file
  --x/xlsx XLSX         
                        XLSX file with labels
```

Example:
```sh
python train.py --train <path_to_images> --val <path_to_images> --use_xlsx --xlsx <path_to_xlsx> 
```
