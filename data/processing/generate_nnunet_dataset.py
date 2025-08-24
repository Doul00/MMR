"""
@brief Script to generate nnUNet dataset from video cases
"""

import os
import sys
import glob
import shutil
import multiprocessing as mp

from os.path import join, basename
from data.definitions import SEG_LABELS
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def get_symlinks_paths(case_dir, train_folder, labels_folder=None):
    """
    Return paths to make symlinks between extracted video files and the nnUnet folders.
    This is to prevent copying a large amount of files.
    """
    if labels_folder is not None:
        label_files = glob.glob(join(case_dir, 'segmentation', '*.png'))
        label_basenames = { basename(x): True for x in label_files }
        rgb_files = [x for x in glob.glob(join(case_dir, 'rgb', '*.png')) if label_basenames.get(basename(x), False)]
    else:
        rgb_files = glob.glob(join(case_dir, 'rgb', '*.png'))

    def _generate_dst_fname(f, is_label):
        frame_num = int(basename(f).split('.')[0])
        return f"{labels_folder}/{basename(case_dir)}_{frame_num}.png" if is_label \
             else f"{train_folder}/{basename(case_dir)}_{frame_num}_0000.png"

    dst_fnames = [_generate_dst_fname(f, False) for f in rgb_files]
    if labels_folder is not None:
        label_dst_fnames = [_generate_dst_fname(f, True) for f in label_files]
        return rgb_files, label_files, dst_fnames, label_dst_fnames
    else:
        return rgb_files, dst_fnames

def create_symlink(src, dst):
    """
    Create a symlink from src to dst.
    """
    if not os.path.exists(dst):
        os.symlink(src, dst)

if __name__ == '__main__':
    root_dir = sys.argv[1]
    test_folder = join(root_dir, 'test')
    train_folder = join(root_dir, 'training_set')

    dataset_name = "Dataset999_SARRARP50"
    output_dir = join(nnUNet_raw, dataset_name)
    imagesTr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagesTs = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelsTr = join(nnUNet_raw, dataset_name, 'labelsTr')
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    symlink_args = []
    num_samples = 0

    for case_dir in glob.glob(join(train_folder, '*')):
        if not os.path.isdir(case_dir):
            continue

        rgb_files, label_files, dst_fnames, label_dst_fnames = get_symlinks_paths(case_dir, imagesTr, labelsTr)
        num_samples += len(rgb_files)
        symlink_args.extend(zip(rgb_files, dst_fnames))
        symlink_args.extend(zip(label_files, label_dst_fnames))

    for case_dir in glob.glob(join(test_folder, '*')):
        rgb_files, dst_fnames = get_symlinks_paths(case_dir, imagesTs)
        num_samples += len(rgb_files)
        symlink_args.extend(zip(rgb_files, dst_fnames))

    print("Number of samples:", num_samples)
    with mp.Pool(16) as pool:
        pool.starmap(create_symlink, symlink_args)

    channels = { 0: 'R', 1: 'G', 2: 'B' }
    generate_dataset_json(output_folder=output_dir,
                          channel_names=channels,
                          labels=SEG_LABELS,
                          num_training_cases=num_samples,
                          file_ending='.png',
                          dataset_name=dataset_name,
    )