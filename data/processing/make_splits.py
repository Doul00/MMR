"""
@brief: Make train/val splits for SARRARP50 dataset.
We make sure that we split by video, not by sample.
"""

import os
import glob
import yaml
import argparse
import random

from typing import List, Tuple
from os.path import join, basename


class VideoDataset:
    """Class representing a surgery video dataset."""

    def __init__(self, root_dirs: List[str]):
        self.root_dirs = root_dirs
        self.seg_samples = []
        self.num_samples = 0
        for root_dir in root_dirs:
            dir_samples = glob.glob(join(root_dir, 'segmentation', '*.png'))
            dir_samples = sorted(dir_samples, key = lambda x: int(basename(x).split('.')[0]))
            self.seg_samples.extend(dir_samples)
            self.num_samples += len(dir_samples)

    def __len__(self):
        return self.num_samples

    def generate_listfile(self) -> List[str]:
        """
        Returns a list of the (segmentation) samples in the video dataset.
        """
        return self.seg_samples


def get_video_datasets(root_dir: str) -> List[VideoDataset]:
    """
    Get video datasets from the root directory.
    """
    video_datasets = []
    for video_dir in glob.glob(join(root_dir, '*')):

        if not os.path.isdir(video_dir):
            continue

        # Videos split in 2 parts
        if len(basename(video_dir).split('_')) != 2:
            video_base_name = "_".join(basename(video_dir).split('_')[:-1])
            subdirs = [join(root_dir, video_base_name + '_' + str(i)) for i in range(1, 3)]
            skip_sample = False
            for x in subdirs:
                if x in added_videos:
                    skip_sample = True
                    break
                added_videos.add(x)

            if skip_sample:
                continue
        else:
            subdirs = [video_dir]
            added_videos.add(video_dir)

        video_datasets.append(VideoDataset(subdirs))
    return video_datasets


def make_splits(video_datasets: List[VideoDataset], val_ratio: float, test_ratio: float) -> Tuple[List[VideoDataset], List[VideoDataset], List[VideoDataset]]: 
    """
    Generate train/val/test video splits.
    """
    total_len = sum(len(vd) for vd in video_datasets)
    num_val_samples = int(total_len * val_ratio)
    num_test_samples = int(total_len * test_ratio)
    random.shuffle(video_datasets)

    video_idx = 0
    collected_val_samples, collected_test_samples = 0, 0
    val_video_datasets, test_video_datasets = [], []
    while collected_val_samples < num_val_samples:
        val_video_datasets.append(video_datasets[video_idx])
        collected_val_samples += len(val_video_datasets[-1])
        video_idx += 1

    while collected_test_samples < num_test_samples:
        test_video_datasets.append(video_datasets[video_idx])
        collected_test_samples += len(test_video_datasets[-1])
        video_idx += 1

    train_video_datasets = video_datasets[video_idx:]

    return train_video_datasets, val_video_datasets, test_video_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make splits for SARRARP50 dataset.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root dir with the dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir for the listfiles.')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of splits to make.')
    parser.add_argument('--val_ratio', type=float, default=0.125, help='Ratio of validation samples.')
    parser.add_argument('--test_ratio', type=float, default=0.125, help='Ratio of test samples.')
    args = parser.parse_args()

    added_videos = set()
    video_datasets = get_video_datasets(args.root_dir)
    with open(join(args.output_dir, 'full_train_list.txt'), 'w') as f:
        for dset in video_datasets:
            listfile = dset.generate_listfile()
            f.write("\n".join(listfile))

    for i in range(args.num_splits):
        print(f"\n\n=== Stats for split {i} ===")
        output_dir = join(args.output_dir, f'split_{i}')
        os.makedirs(output_dir, exist_ok=True)
        split_dsets = make_splits(video_datasets, args.val_ratio, args.test_ratio)

        for video_splits, split_name in zip(split_dsets, ['train', 'val', 'test']):
            num_samples = 0
            listfile_path = join(output_dir, f'{split_name}.txt')
            with open(listfile_path, 'w') as f:
                split_samples = []
                for dset in video_splits:
                    num_samples += len(dset)
                    listfile = dset.generate_listfile()
                    split_samples.extend(listfile)
                f.write("\n".join(split_samples))

            print(f"Number of samples in {split_name}: {num_samples}")
