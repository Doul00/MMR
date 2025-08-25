#!/bin/bash

#######################
# Evaluate the model results
# $PREDICTIONS_DIR and DATA_DIR follow the format: 
# pred_root_dir
# |-- video_41
# |   |-- segmentation
# |   |   |--- 000000xxxxx.png
#
#######################

EXP_NAME=$1
DATA_DIR=/nfs/home/aamadou/Data/SAR_RARP50/raw/test
PREDICTIONS_DIR=/nfs/home/aamadou/Projects/MMR/output/logs/${EXP_NAME}/weights/predictions
SAR_RARP50_REPO_DIR=/nfs/home/aamadou/Projects/SAR_RARP50-evaluation

cd $SAR_RARP50_REPO_DIR && python -m scripts.sarrarp50 evaluate $DATA_DIR $PREDICTIONS_DIR --ignore_actions