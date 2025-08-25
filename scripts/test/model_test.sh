#!/bin/bash

#######################
# Generate predictions for the Mask2Former model
# They will be saved in the checkpoint directory
#######################

ROOT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

FOLD=$1
LOG_DIR=/nfs/home/aamadou/Projects/MMR/output/logs
TEST_LISTFILE=$ROOT_DIR/data/splits_final/test.txt
CHECKPOINT_PATH=/nfs/home/aamadou/Projects/MMR/output/logs/train-m2f-full-aug-fold-${FOLD}/weights/best.ckpt

# Compile and install custom DeformableAttention CUDA kernel
# You can comment it if done once - left uncommented as this needs to be done everytime
# when using docker containers
cd $ROOT_DIR/model/modeling/pixel_decoder/ops && sh make.sh

python $ROOT_DIR/test.py \
    hydra.output_subdir=null \
    hydra.run.dir=. \
    +ckpt_path=${CHECKPOINT_PATH} \
    testing_opts.save_dir=$(dirname ${CHECKPOINT_PATH}) \
    testing_opts.test_minibatch_size=32 \
    data_opts.test_dataset.listfile=$TEST_LISTFILE
