#!/bin/bash

CURR_PATH=$(realpath $0)
ROOT_DIR=$(dirname $(dirname $(dirname $CURR_PATH)))
ROOT_DATA_DIR=/nfs/home/aamadou/Data/SAR_RARP50/raw/training_set
export PYTHONPATH=$ROOT_DIR

python $ROOT_DIR/data/processing/make_splits.py \
    --root_dir /nfs/home/aamadou/Data/SAR_RARP50/raw/test \
    --output_dir $ROOT_DIR/data/test_splits \
    --num_splits 1 \
    --val_ratio 0 \
    --test_ratio 0
