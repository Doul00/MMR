"""
@brief Copies the nnUnet predictions into the format expected by the eval code
"""

import os
import sys
import glob
import numpy as np
import multiprocessing as mp
from PIL import Image

from os.path import join, basename, dirname

def copy_file_to_folder(file_path, output_folder):
    file_path_no_ext = basename(file_path).split('.')[0]
    splits = file_path_no_ext.split('_')
    video_name = f"{splits[0]}_{splits[1]}"
    frame_num = splits[2].zfill(9)

    # Image needs to be re-saved in RGB
    output_path = join(output_folder, video_name, 'segmentation', f"{frame_num}.png")
    os.makedirs(dirname(output_path), exist_ok=True)
    Image.open(file_path).convert('RGB').save(output_path)


if __name__ == '__main__':
    nnunet_predictions_dir = sys.argv[1]
    output_folder = dirname(nnunet_predictions_dir) + "/nnunet_predict_formatted"
    os.makedirs(output_folder, exist_ok=True)

    all_preds = glob.glob(join(nnunet_predictions_dir, '*.png'))
    pool_args = [(p, output_folder) for p in all_preds]
    with mp.Pool(32) as pool:
        pool.starmap(copy_file_to_folder, pool_args)
