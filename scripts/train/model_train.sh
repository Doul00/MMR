#!/bin/bash

#######################
# Train the Mask2Former model
# Results will be saved user $LOG_DIR/$EXP_NAME
#######################

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

LOG_DIR=/nfs/home/aamadou/Projects/MMR/output/logs

FOLD_NUM=$1
FOLD_TRAIN_FILE=$ROOT_DIR/data/splits_final/fold_${FOLD_NUM}/train.txt
FOLD_VAL_FILE=$ROOT_DIR/data/splits_final/fold_${FOLD_NUM}/val.txt

EXP_NAME=train-m2f-full-aug-fold-${FOLD_NUM}

# Compile and install custom DeformableAttention CUDA kernel
# You can comment it if done once - left uncommented as this needs to be done everytime
# when using docker containers
cd $ROOT_DIR/model/modeling/pixel_decoder/ops && sh make.sh

python $ROOT_DIR/train.py \
    hydra.output_subdir=null \
    hydra.run.dir=. \
    exp_name=$EXP_NAME \
    log_dir=$LOG_DIR \
    data_opts.train_dataset.listfile=$FOLD_TRAIN_FILE \
    data_opts.val_dataset.listfile=$FOLD_VAL_FILE
    # data_opts.train_dataset.listfile=$DEBUG_TRAIN_FILE \
    # data_opts.val_dataset.listfile=$DEBUG_VAL_FILE
