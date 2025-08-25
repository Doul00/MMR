#!/bin/bash

#######################
# Merge predictions from multiple models
#######################

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
EXP_NAME=train-m2f-full-aug-ensemble
OUTPUT_DIR=/nfs/home/aamadou/Projects/MMR/output/test_results/$EXP_NAME/predictions

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
ROOT_DIRS=(
    /nfs/home/aamadou/Projects/MMR/output/logs/train-m2f-full-aug-fold-0/weights/predictions
    /nfs/home/aamadou/Projects/MMR/output/logs/train-m2f-full-aug-fold-1/weights/predictions
    /nfs/home/aamadou/Projects/MMR/output/logs/train-m2f-full-aug-fold-2/weights/predictions
    /nfs/home/aamadou/Projects/MMR/output/logs/train-m2f-full-aug-fold-3/weights/predictions
    /nfs/home/aamadou/Projects/MMR/output/logs/train-m2f-full-aug-fold-4/weights/predictions
)

python $ROOT_DIR/helpers/merge_predictions.py \
    --root_dirs ${ROOT_DIRS[@]} \
    --output_dir $OUTPUT_DIR