"""
@brief: Dataset utils for SARRARP50 dataset
"""

import torch
import numpy as np
import pytorch_lightning as pl

from PIL import Image
from collections import defaultdict
from os.path import basename, dirname
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from helpers.image import overlay_segmentation
from data.transforms import get_transforms


class SARRARP50Dataset(Dataset):
    def __init__(self, config: dict, is_test: bool = False, **kwargs):
        """
        Dataloader for SARRARP50 dataset.

        Args:
            config (dict): Dataset config, contains the following fields:
                listfile (str): List of (segmentation) samples used by the dataset. 
                img_transforms (list): List of image transforms.
                label_transforms (list): List of label transforms.
            is_test (bool): Whether the dataset is for testing - Changes the way the index is loaded since we send the full video frames
        """
        self.config = config
        self.img_transforms = get_transforms(self.config['img_transforms'])
        self.is_test = is_test
        if self.is_test:
            # Index only stores segmentation paths
            self.index = self._make_index_test(self.config['listfile'])
        else:
            self.index = self._make_index_train(self.config['listfile'])

    def _make_index_test(self, listfile: str):
        all_samples = [x.strip() for x in open(listfile, 'r').readlines()]
        videos_to_samples = defaultdict(list)
        for sample in all_samples:
            video_name = basename(dirname(dirname(sample)))
            videos_to_samples[video_name].append(sample)

        # Sort
        for video_name, samples in videos_to_samples.items():
            videos_to_samples[video_name] = sorted(samples, key=lambda x: int(basename(x).split('.')[0]))

        return videos_to_samples

    def _make_index_train(self, listfile: str):
        all_samples = [x.strip() for x in open(listfile, 'r').readlines()]
        index = []
        for sample in all_samples:
            img_path = f"{dirname(dirname(sample))}/rgb/{basename(sample)}"
            index.append((img_path, sample))
        return index

    def _train_getitem(self, index):

        img, label = self.index[index]
        img = Image.open(img)
        label = Image.open(label)

        img, label = self.img_transforms(img, label)
        # Remove channel dimension
        label = label[0].unsqueeze(0)
        return {
            'images': img,
            'masks': label
        }

    def _test_getitem(self, index):
        video_name = list(self.index.keys())[index]
        video_samples = self.index[video_name]
        images, masks = [], []
        for vs in video_samples:
            img_path = f"{dirname(dirname(vs))}/rgb/{basename(vs)}"
            img = Image.open(img_path)
            mask = Image.open(vs)

            img, mask = self.img_transforms(img, mask)
            mask = mask[0].unsqueeze(0)
            images.append(img)
            masks.append(mask)

        images = torch.stack(images)
        masks = torch.stack(masks)
        return {
            'mask_names': video_samples,
            'images': images,
            'masks': masks
        }

    def __getitem__(self, index):
        if self.is_test:
            return self._test_getitem(index)
        else:
            return self._train_getitem(index)

    def __len__(self):
        return len(self.index)


class SARRARP50DataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def train_dataloader(self):
        train_dataset_cfg = self.config['data_opts']['train_dataset']
        dataset = SARRARP50Dataset(config=train_dataset_cfg)
        return DataLoader(dataset,
            batch_size=train_dataset_cfg['batch_size'],
            shuffle=True,
            num_workers=train_dataset_cfg['num_workers'],
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        val_dataset_cfg = self.config['data_opts']['val_dataset']
        dataset = SARRARP50Dataset(config=val_dataset_cfg)
        return DataLoader(dataset,
            batch_size=val_dataset_cfg['batch_size'],
            shuffle=False,
            num_workers=val_dataset_cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        if 'test_dataset' not in self.config['data_opts']:
            return None

        test_dataset_cfg = self.config['data_opts']['test_dataset']
        dataset = SARRARP50Dataset(config=test_dataset_cfg, is_test=True)
        return DataLoader(dataset,
            batch_size=1,
            shuffle=False,
            num_workers=test_dataset_cfg['num_workers'],
            pin_memory=False,
            drop_last=False
        )