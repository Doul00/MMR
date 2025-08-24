#!/bin/bash

#######################
# Train the nnUNet model
#######################
FOLD=0 # Choose in [0 - 4]
NNUNET_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/nnUNet
export nnUNet_raw=$NNUNET_ROOT/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_preprocessed
export nnUNet_results=$NNUNET_ROOT/nnUNet_results

nnUNetv2_train 999 2d $FOLD --npz -num_gpus 2