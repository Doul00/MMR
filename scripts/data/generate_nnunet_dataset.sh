#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
export PYTHONPATH=$ROOT_DIR


DATA_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/raw
NNUNET_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/nnUNet
export nnUNet_raw=$NNUNET_ROOT/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_preprocessed
export nnUNet_results=$NNUNET_ROOT/nnUNet_results

python $ROOT_DIR/data/processing/generate_nnunet_dataset.py $DATA_ROOT