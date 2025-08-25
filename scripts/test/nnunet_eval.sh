#!/bin/bash

#######################
# Generate predictions for the nnUNet models (with ensembling)
#######################

NNUNET_ROOT=/nfs/home/aamadou/Data/SAR_RARP50/nnUNet
export nnUNet_raw=$NNUNET_ROOT/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_ROOT/nnUNet_preprocessed
export nnUNet_results=$NNUNET_ROOT/nnUNet_results

INPUT_FOLDER=/nfs/home/aamadou/Data/nnUnet_imagesTs_rest
OUTPUT_FOLDER=/nfs/home/aamadou/Projects/MMR/output/test_results/nnunet_predict
OUTPUT_FOLDER_PP=/nfs/home/aamadou/Projects/MMR/output/test_results/nnunet_predict_pp

echo "Predicting..."
nnUNetv2_predict -d Dataset999_SARRARP50 -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans -npp 16 -nps 16

echo "Prediction done! Applying postprocessing..."
nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP \
 -pp_pkl_file /nfs/home/aamadou/Data/SAR_RARP50/nnUNet/nnUNet_results/Dataset999_SARRARP50/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
 -np 32 \
 -plans_json /nfs/home/aamadou/Data/SAR_RARP50/nnUNet/nnUNet_results/Dataset999_SARRARP50/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json