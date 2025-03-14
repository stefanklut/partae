#!/bin/bash

TRAIN_DIRS="/data/data/ovdr/2.09.09_ushmm_htr/*/ /data/data/ovdr/triado/*/"
XLSX_FILE="./Documentsegmentatie_GT.xlsx"
OUTPUT_DIR=./output

checkpoint_dir=$(mktemp -d -p .)

# Run training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --output $checkpoint_dir \
    --train $TRAIN_DIRS \
    --xlsx $XLSX_FILE \
    --dropout 0.5 \
    --label_smoothing 0.2 \
    --randomize_document_order \
    --sample_same_inventory \
    --number_of_images 3 \
    --batch_size 8 \
    --epochs 40 \
    --num_workers 4 \
    --unfreeze_imagenet 0.5 \
    --unfreeze_roberta 0.5 \

# Detemine the best checkpoint
best_checkpoint=$(find $checkpoint_dir -name "*val_acc=*.ckpt" | \
    xargs -I {} basename {} | \
    sort -t '=' -k 3 -nr | \
    head -n 1)

best_checkpoint=$checkpoint_dir/version_0/checkpoints/$best_checkpoint

# Run inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --output $OUTPUT_DIR \
    --input $INPUT_DIRS \
    --checkpoint $best_checkpoint
