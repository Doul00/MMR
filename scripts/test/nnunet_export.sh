#!/bin/bash

#######################
# Export nnUnet models
#######################

NNUNET_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/nnUNet
export nnUNet_raw=$NNUNET_ROOT/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_preprocessed
export nnUNet_results=$NNUNET_ROOT/nnUNet_results

ZIP_NAME=nnunet_model.zip

nnUNetv2_export_model_to_zip \
  -d 999 \
  -o $ZIP_NAME \
  -c 2d \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  -chk checkpoint_best.pth