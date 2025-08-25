"""
@brief: Ensemble of Mask2Former predictions
"""

import os
import glob
import argparse
import numpy as np
from os.path import basename
from multiprocessing import Pool
from PIL import Image


from data.definitions import SEG_LABELS

def majority_vote(seg_masks: np.ndarray) -> np.ndarray:
    """
    Perform majority voting on segmentation masks.
    Args:
        seg_masks: shape (N, H, W) 
    """
    N, H, W = seg_masks.shape
    seg_masks_flat = seg_masks.reshape(N, -1)

    num_classes = len(SEG_LABELS)
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=num_classes),
        axis=0,
        arr=seg_masks_flat
    )  # shape (H*W, num_classes)

    # Take argmax (most common label per pixel)
    majority = counts.argmax(axis=1)
    majority = majority.reshape(H, W)[..., None]
    majority = np.repeat(majority, 3, axis=-1)
    return majority


def ensemble_predictions(subpath, root_dirs, output_dir):
    """
    Merges the predictions from multiple models.
    Args:
        subpath: path to the prediction file
        root_dirs: list of root directories containing the prediction files
        output_dir: directory to save the ensemble prediction
    """
    pred_files = [os.path.join(root_dir, subpath) for root_dir in root_dirs]
    pred_masks = [np.array(Image.open(pred_file))[..., 0] for pred_file in pred_files]
    pred_masks = np.stack(pred_masks).astype(np.uint8)
    ensemble_mask = majority_vote(pred_masks)
    dst_file = os.path.join(output_dir, subpath)
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    Image.fromarray(ensemble_mask).save(dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirs", nargs='+', required=True, help="Root dirs containing the 'prediction' folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the videos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pred_files = [os.path.join(root_dir, f) for root_dir, _, files in os.walk(args.root_dir[0])
                                        for f in files if f.endswith('.png')]
    subpaths = [f.replace(args.root_dirs[0], '') for f in pred_files]
    pool_args = [(subpath, args.root_dirs, args.output_dir) for subpath in subpaths]
    with Pool(processes=32) as pool:
        pool.starmap(ensemble_predictions, pool_args)
