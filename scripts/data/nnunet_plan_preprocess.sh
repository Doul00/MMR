#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(pwd)))/code
export PYTHONPATH=$ROOT_DIR


NNUNET_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/nnUNet
export nnUNet_raw=$NNUNET_ROOT/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_preprocessed
export nnUNet_results=$NNUNET_ROOT/nnUNet_results

nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity -c 2d