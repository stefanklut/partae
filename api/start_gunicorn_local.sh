#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to partae base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

PARTAE_MAX_QUEUE_SIZE=128 \
PARTAE_MODEL_BASE_PATH="/home/stefan/Downloads/" \
PARTAE_OUTPUT_BASE_PATH="/tmp/" \

mkdir -p $PARTAE_OUTPUT_BASE_PATH

export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=1
export GUNICORN_ACCESSLOG='-'

python api/gunicorn_app.py