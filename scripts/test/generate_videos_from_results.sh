#!/bin/bash

#######################
# Generate videos from the model results
# Expects the results directory to be in the format:
# pred_root_dir
# |-- video_41
# |   |-- segmentation
# |   |   |--- 000000xxxxx.png
#######################

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
EXP_NAME=train-m2f-full-no-aug
OUTPUT_DIR=/nfs/home/aamadou/Projects/MMR/output/test_results/$EXP_NAME/videos

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

python $ROOT_DIR/helpers/generate_videos_from_results.py \
    --gt_root_dir /nfs/home/aamadou/Data/SAR_RARP50/raw/test \
    --pred_root_dir /nfs/home/aamadou/Projects/MMR/output/logs/${EXP_NAME}/weights/predictions\
    --output_dir $OUTPUT_DIR