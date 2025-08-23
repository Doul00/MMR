#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

FOLD=0
SPLITS_DIR=$ROOT_DIR/data/splits
LOG_DIR=/nfs/home/aamadou/Projects/MMR/output/logs

python $ROOT_DIR/train.py \
    exp_name=mvit_fold_$FOLD \
    log_dir=$LOG_DIR \
    data_opts.train_dataset.listfile=$SPLITS_DIR/split_$FOLD/train.txt \
    data_opts.val_dataset.listfile=$SPLITS_DIR/split_$FOLD/val.txt