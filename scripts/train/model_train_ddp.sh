#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

SPLITS_DIR=$ROOT_DIR/data/splits
LOG_DIR=/nfs/home/aamadou/Projects/MMR/output/logs

DEBUG_TRAIN_FILE=$SPLITS_DIR/debug/train.txt
DEBUG_VAL_FILE=$SPLITS_DIR/debug/val.txt

FOLD_TRAIN_FILE=$ROOT_DIR/data/splits_final/train.txt
FOLD_VAL_FILE=$ROOT_DIR/data/splits_final/val.txt

EXP_NAME=$2

# Compile and install custom DeformableAttention CUDA kernel
# You can comment it if done once - left uncommented as this needs to be done everytime
# when using docker containers
cd $ROOT_DIR/model/modeling/pixel_decoder/ops && sh make.sh

GPUS_PER_NODE=8
NODE_RANK=$1
MASTER_PORT=29500
export NUM_NODES=2

MASTER_IP_FILE=$ROOT_DIR/../output/master_ip.txt
echo "WITH PROCESS AT $NODE_RANK"

if [ $NODE_RANK -eq 0 ]; then
    MASTER_ADDR=$(hostname -i)
    echo $MASTER_ADDR > $MASTER_IP_FILE
    echo "MASTER_ADDR: $MASTER_ADDR"
else
    while [ ! -f $MASTER_IP_FILE ]; do sleep 1; done
    MASTER_ADDR=$(cat $MASTER_IP_FILE)
    echo "READ MASTER_ADDR: $MASTER_ADDR"
fi

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $ROOT_DIR/train.py \
    hydra.output_subdir=null \
    hydra.run.dir=. \
    exp_name=$EXP_NAME \
    log_dir=$LOG_DIR \
    data_opts.train_dataset.listfile=$FOLD_TRAIN_FILE \
    data_opts.val_dataset.listfile=$FOLD_VAL_FILE

    # data_opts.train_dataset.listfile=$DEBUG_TRAIN_FILE \
    # data_opts.val_dataset.listfile=$DEBUG_VAL_FILE

