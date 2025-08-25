"""
@brief: Generate videos from the results of a given model 
"""

import os
import glob
import torch
import argparse
import numpy as np
from os.path import basename
from PIL import Image
from moviepy import ImageSequenceClip
from multiprocessing import Pool

from data.definitions import SEG_LABELS
from helpers.image import overlay_segmentation


def generate_overlay(img_path, mask_path, pred_path):
    img, mask, pred = [np.array(Image.open(x)).transpose(2, 0, 1) for x in [img_path, mask_path, pred_path]]
    img, mask, pred = [torch.from_numpy(x) for x in [img, mask, pred]]
    overlay_img = overlay_segmentation(img, mask, pred, len(SEG_LABELS))
    return overlay_img


def generate_video_for_folder(imgs_dir, gt_dir, pred_dir, output_dir, fps):
    def _sort_files(files):
        return sorted(files, key=lambda x: int(basename(x).split(".")[0]))

    os.makedirs(output_dir, exist_ok=True)
    gt_seg, pred_seg = [_sort_files(glob.glob(os.path.join(x, '*.png'))) for x in [gt_dir, pred_dir]]
    assert len(gt_seg) == len(pred_seg), f"Incomplete predictions for video at path {imgs_dir}"
    imgs = [os.path.join(imgs_dir, basename(x)) for x in gt_seg]

    # High-res overlay generation is slow
    with Pool(processes=32) as pool:
        output_overlays = pool.starmap(generate_overlay, zip(imgs, gt_seg, pred_seg))

    video_fname = os.path.join(output_dir, "video.mp4")
    if os.path.exists(video_fname):
        os.remove(video_fname)
    ImageSequenceClip(output_overlays, fps=fps).write_videofile(video_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root_dir", type=str, required=True, help="Test directory containing video folders, each with their `segmentation` subfolder")
    parser.add_argument("--pred_root_dir", type=str, required=True, help="Test directory containing video folders, each with their `segmentation` subfolder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the videos")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the output video")
    args = parser.parse_args()

    gt_root_dir = args.gt_root_dir
    pred_root_dir = args.pred_root_dir
    os.makedirs(args.output_dir, exist_ok=True)

    for video_name in os.listdir(pred_root_dir):
        imgs_dir = os.path.join(gt_root_dir, video_name, 'rgb')
        gt_dir = os.path.join(gt_root_dir, video_name, 'segmentation')
        pred_dir = os.path.join(pred_root_dir, video_name, 'segmentation')
        output_dir = os.path.join(args.output_dir, video_name)
        generate_video_for_folder(imgs_dir, gt_dir, pred_dir, output_dir, args.fps)
