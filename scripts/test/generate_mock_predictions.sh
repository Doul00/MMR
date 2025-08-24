#!/bin/bash

DATA_DIR=/nfs/home/aamadou/Data/SAR_RARP50/raw/test
MOCK_PREDICTIONS_DIR=/nfs/home/aamadou/Data/SAR_RARP50/mock_predictions
SAR_RARP50_REPO_DIR=/nfs/home/aamadou/Projects/SAR_RARP50-evaluation

cd $SAR_RARP50_REPO_DIR && python -m scripts.sarrarp50 generate $DATA_DIR $MOCK_PREDICTIONS_DIR --overwrite