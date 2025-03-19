#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to partae base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

PARTAE_MAX_QUEUE_SIZE=128 \
PARTAE_MODEL_BASE_PATH="/home/stefan/Downloads/" \
PARTAE_OUTPUT_BASE_PATH="/tmp/" \
FLASK_DEBUG=true FLASK_APP=api.flask_app.py flask run

mkdir -p $PARTAE_OUTPUT_BASE_PATH