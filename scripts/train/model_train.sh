#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

SPLITS_DIR=$ROOT_DIR/data/splits
LOG_DIR=/nfs/home/aamadou/Projects/MMR/output/logs

DEBUG_TRAIN_FILE=$SPLITS_DIR/debug/train.txt
DEBUG_VAL_FILE=$SPLITS_DIR/debug/val.txt

FOLD_TRAIN_FILE=$ROOT_DIR/data/splits_final/train.txt
FOLD_VAL_FILE=$ROOT_DIR/data/splits_final/val.txt

EXP_NAME=m2f_debug_new

cd $ROOT_DIR/model/modeling/pixel_decoder/ops && sh make.sh

python $ROOT_DIR/train.py \
    hydra.output_subdir=null \
    hydra.run.dir=. \
    exp_name=$EXP_NAME \
    log_dir=$LOG_DIR \
    data_opts.train_dataset.listfile=$DEBUG_TRAIN_FILE \
    data_opts.val_dataset.listfile=$DEBUG_VAL_FILE
    # data_opts.train_dataset.listfile=$FOLD_TRAIN_FILE \
    # data_opts.val_dataset.listfile=$FOLD_VAL_FILE

