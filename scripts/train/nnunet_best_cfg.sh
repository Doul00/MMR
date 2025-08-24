#!/bin/bash

NNUNET_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/nnUNet
export nnUNet_raw=$NNUNET_ROOT/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_preprocessed
export nnUNet_results=$NNUNET_ROOT/nnUNet_results

# for FOLD in {0..4}; do
#     nnUNetv2_train 999 2d $FOLD --val_best --npz
# done
nnUNetv2_find_best_configuration 999 -c 2d