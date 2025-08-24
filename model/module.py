"""
@brief: ToolSegmenterModule pytorch lightning module. Handles training loop and inference
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl
import multiprocessing as mp

from os.path import basename, dirname
from PIL import Image
from typing import Dict, Any, Tuple
from omegaconf import DictConfig
from torch.nn import functional as F
from detectron2.structures import Instances
from moviepy import ImageSequenceClip
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, SequentialLR

from data.definitions import SEG_LABELS
from model.model import ToolSegmenterModel
from helpers.image import overlay_segmentation


def linear_warmup_cosine_annealing(optimizer, max_steps: int, warmup_steps: int, eta_min: float =0):

    def warmup_fn(step):
        if step >= warmup_steps:
            return 1.0
        return step / warmup_steps

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=eta_min)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler


class ToolSegmenterModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = ToolSegmenterModel(self.config['model_opts'])
        training_opts = self.config['training_opts']
        self.lr = training_opts['learning_rate']
        self.optimizer = training_opts['optimizer']
        self.scheduler = training_opts['scheduler']

        self._test_samples = []
        self._test_results = []

        if 'testing_opts' in self.config:
            self._test_minibatch_size = self.config['testing_opts']['test_minibatch_size']
            self._test_final_image_size_hw = self.config['testing_opts']['final_image_size_hw']
        else:
            self._test_minibatch_size = 32
            self._test_final_image_size_hw = [1080, 1920]

    def _targets_to_instances(self, mask: torch.Tensor) -> Instances:
        """
        Given a segmentation mask, create binary masks for each class and store it into an Instances object.
        Args:
            mask: a tensor of shape (1, H, W) containing the labels.
        Returns:
            an Instances object
        """
        instances = Instances(image_size=(mask.shape[-2], mask.shape[-1]))
        instances.gt_classes = torch.LongTensor(list(SEG_LABELS.values())).to(mask.device)
        bit_mask_tensor = torch.zeros((len(instances.gt_classes), mask.shape[-2], mask.shape[-1])).to(mask.device)
        for val in instances.gt_classes:
            bit_mask = torch.zeros_like(mask)
            bit_mask[mask == val] = 1
            bit_mask_tensor[val] = bit_mask
        instances.gt_masks = bit_mask_tensor
        return instances

    def _compute_loss(self, batch, batch_idx) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Computes the multi-scale loss for a batch of images and masks.
        Args:
            batch: a dict containing the following keys:
                "images": a tensor of shape (B, C, H, W) containing the images
                "masks": a tensor of shape (B, 1, H, W) containing the masks
        Returns:
            a dict containing the losses and the outputs
        """

        imgs, targets = batch['images'], batch['masks']
        outputs = self.model(imgs)
        targets = [self._targets_to_instances(target) for target in targets]
        targets = self.model.prepare_targets(targets, imgs)
        losses = self.model.criterion(outputs, targets)
        for k in list(losses.keys()):
            if k in self.model.criterion.weight_dict:
                losses[k] *= self.model.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses, outputs

    def training_step(self, batch, batch_idx):
        """Runs a training step for a batch of images and masks."""

        losses, _ = self._compute_loss(batch, batch_idx)
        total_loss = torch.stack(list(losses.values())).sum()
        for k, v in losses.items():
            self.log(f'train_{k}', v, prog_bar=False, logger=True)
        self.log('train_loss', total_loss, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Runs a validation step for a batch of images and masks and optionally saves predictions to tensorboard."""

        imgs, masks = batch['images'], batch['masks']
        losses, outputs = self._compute_loss(batch, batch_idx)
        total_loss = torch.stack(list(losses.values())).sum()
        for k, v in losses.items():
            self.log(f'train_{k}', v, prog_bar=False, logger=True)
        self.log('val_loss', total_loss, prog_bar=True, logger=True)

        # Log results to tensorboard for visualization
        if batch_idx % 10 == 0:
            processed_results = self._postprocess_outputs(imgs, outputs)
            processed_results = torch.stack([result.argmax(dim=0) for result in processed_results])
            rand_indexes = torch.randint(0, len(processed_results), (3,))
            all_overlays = []
            for i in rand_indexes:
                rand_mask = masks[i].squeeze()
                rand_pred = processed_results[i]
                rand_img = imgs[i]
                overlay_img = overlay_segmentation(rand_img, rand_mask, rand_pred, len(SEG_LABELS))
                all_overlays.append(overlay_img)

            all_overlays = np.concatenate(all_overlays, axis=0)
            self.logger.experiment.add_images(f"val_overlay", all_overlays, self.global_step, dataformats='HWC')

        return total_loss


    def _save_video_predictions(self, imgs, masks, preds, mask_names, predictions_dir, visualizations_dir):
        """
        Saves the predictions for a video.
        Args:
            imgs: a tensor of shape (N, C, H, W) containing the images
            masks: a tensor of shape (N, 1, H, W) containing the masks
            preds: a tensor of shape (N, H, W) containing the predictions
            mask_names: a list of strings containing the names of the masks
            predictions_dir: a string containing the directory to save the predictions
        """
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)

        # Do final predictions upsampling
        preds = preds.unsqueeze(1).to(torch.uint8)
        pred_upsampled = F.interpolate(preds, size=(self._test_final_image_size_hw[0], self._test_final_image_size_hw[1]), mode="nearest-exact")
        pred_upsampled = pred_upsampled.cpu().numpy()
        for i in range(pred_upsampled.shape[0]):
            pred_upsampled_i = np.repeat(pred_upsampled[i], 3, axis=0)
            pred_upsampled_i = pred_upsampled_i.transpose(1, 2, 0)
            Image.fromarray(pred_upsampled_i).save(os.path.join(predictions_dir, basename(mask_names[i])))


    def test_step(self, batch, batch_idx):
        """
        Runs prediction on a list of ordered frames of a video.
        Args:
            batch: a dict containing the following keys:
                "video_name": a string containing the name of the video
                "images": a tensor of shape (N, C, H, W) containing the images
                "masks": a tensor of shape (N, 1, H, W) containing the masks
        """
        imgs = batch['images'][0] # PL lightning returns a 5D tensor here due to the batch size of 1
        num_batches = imgs.shape[0] // self._test_minibatch_size
        if imgs.shape[0] % self._test_minibatch_size != 0:
            num_batches += 1

        outputs = []
        for i in range(num_batches):
            imgs_batch = imgs[i * self._test_minibatch_size:(i + 1) * self._test_minibatch_size]
            b_out = self.model(imgs_batch)
            outputs.append(b_out)

        # Merge output dicts
        merged_outputs = {}
        for ok in ['pred_logits', 'pred_masks']:
            merged_outputs[ok] = torch.cat([o[ok] for o in outputs], dim=0)

        processed_results = self._postprocess_outputs(imgs, merged_outputs)
        processed_results = torch.stack([result.argmax(dim=0) for result in processed_results])
        self._test_results.append(processed_results)
        self._test_samples.append(batch)

    def on_test_epoch_end(self):
        """Processes the test results and saves the predictions and visualizations."""

        logger = self.logger
        log_dir = logger.log_dir
        preds_dir = log_dir + "/test_results/predictions"
        viz_dir = log_dir + "/test_results/visualizations"
        os.makedirs(preds_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        pool_args = []
        for i in range(len(self._test_samples)):
            batch = self._test_samples[i]
            imgs = batch['images'][0]
            masks = batch['masks'][0]
            mask_names = [y for x in batch['mask_names'] for y in x]
            video_name = basename(dirname(dirname(mask_names[0])))
            preds = self._test_results[i]

            preds_dir = os.path.join(preds_dir, video_name, 'segmentation')
            viz_dir = os.path.join(viz_dir, video_name)
            pool_args.append((imgs, masks, preds, mask_names, preds_dir, viz_dir))

        # self._save_video_predictions(*pool_args[0])
        with mp.Pool(processes=8) as pool:
            pool.map(self._save_video_predictions, pool_args)

    def _postprocess_outputs(self, imgs: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            imgs: a tensor of shape (B, C, H, W)
            outputs: a dict containing the following keys:
                "pred_logits": a tensor of shape (B, Q, K+1)
                "pred_masks": a tensor of shape (B, K, H, W)
                With Q the number of queries and K+1 the number of classes + no object class
        Returns:
            a tensor of shape (B, K, H, W) containing the logits of the segmentation masks
        """
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(imgs.shape[-2], imgs.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):

            r = self.model.semantic_inference(mask_cls_result, mask_pred_result)
            processed_results.append(r)

        return processed_results


    def configure_optimizers(self) -> Dict[str, Any]:
        if self.config['training_opts']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.config['training_opts']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.config['training_opts']['optimizer']} not supported")

        result_dict = {'optimizer': optimizer}
        if 'scheduler' in self.config['training_opts']:
            if self.config['training_opts']['scheduler'] == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=self.config['training_opts']['max_epochs'], eta_min=1e-7)
            elif self.config['training_opts']['scheduler'] == 'plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            elif self.config['training_opts']['scheduler'] == 'linear_warmup_cosine_annealing':
                scheduler = linear_warmup_cosine_annealing(optimizer, max_steps=self.config['training_opts']['max_epochs'], warmup_steps=20, eta_min=1e-7)
            else:
                raise ValueError(f"Scheduler {self.config['training_opts']['scheduler']} not supported")
            result_dict['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            }

        return result_dict