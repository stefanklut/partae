# Partae

This is a tool to find what scans belong to the same document. It uses AI to compare the images and the text of consecutive scans and determine if they belong to the same document. It classifies a scan as either the start of a document or a continuation of the previous document. 

## Table of Contents
- [Partae](#partae)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Conda](#conda)
    - [Docker](#docker)
      - [Manual Installation](#manual-installation)
  - [Training](#training)
  - [Inference](#inference)
    - [Docker API](#docker-api)
  - [Contact](#contact)
    - [Issues](#issues)
    - [Contributions](#contributions)


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

First copy the Partae directory to the temporary docker directory:
```sh
tmp_dir=$(mktemp -d)
cp -r -T <PATH_TO_DIR> $tmp_dir/partae
cp Dockerfile $tmp_dir/Dockerfile
cp _entrypoint.sh $tmp_dir/_entrypoint.sh
cp .dockerignore $tmp_dir/.dockerignore
```

Then build the docker image using the following command:
```sh
docker build -t docker.partae $tmp_dir
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
## Inference

### Docker API

To use the docker image as an API service, we recommend using docker compose. The docker compose file is provided in the [`docker-compose.yml`][docker_compose_link] file. The docker compose file can be run using the following command:
```sh
docker-compose up
```

Then request the API (in this example using curl) with the following command:
```sh
curl -X POST 0.0.0.0:5000/predict \
  -F identifier=<unique_id> \
  -F model=</path/relative/to/model_base_path.pt> \
  -F images[]=@<image_path1> -F images[]=@<image_path2> -F images[]=@<image_path3 \
  -F texts[]=@<pagexml_path1> -F texts[]=@<pagexml_path2> -F texts[]=@<pagexml_path3>
```
Ensure that the image paths and the pagexml paths belong to the same document. And that they are subsequent pages in the directory. To send an empty image, for example before the first image, you can specify the path to the image as `null`. The identifier is a unique identifier for the document, the output will be saved in a directory with the name being the unique id. The model is the path to the model file relative to the model base path. The model base path is the path to the directory where the model is stored. The model base path is set in the [`docker-compose.yml`][docker_compose_link] file. For example, the base path is set to `/models` and the model is stored in `/models/version1/checkpoints/model.pt` then the model path is `version1/checkpoints/model.pt`.

## Contact
This project was made while working at the [KNAW Humanities Cluster Digital Infrastructure][huc_di_link]
### Issues
Please report any bugs or errors that you find to the [issues][issues_link] page, so that they can be looked into. Try to see if an issue with the same problem/bug is not still open. Feature requests should also be done through the [issues][issues_link] page.

### Contributions
If you discover a bug or missing feature that you would like to help with please feel free to send a [pull request][pull_request_link]. 


<!-- Images and Links Shorthand-->
[conda_install_link]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
[mamba_install_link]: https://mamba.readthedocs.io/en/latest/installation.html
[docker_install_link]: https://docs.docker.com/engine/install/
[huc_di_link]: https://di.huc.knaw.nl/
[environment_link]: environment.yml
[docker_compose_link]: docker/docker-compose.yml
[issues_link]: https://github.com/stefanklut/partae/issues