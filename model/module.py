"""
@brief: ToolSegmenterModule pytorch lightning module. Handles training loop and inference
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl

from os.path import basename, dirname
from multiprocessing import Pool
from PIL import Image
from typing import Dict, Any, Tuple, List
from omegaconf import DictConfig
from torch.nn import functional as F
from detectron2.structures import Instances
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from concurrent.futures import ThreadPoolExecutor

from data.definitions import SEG_LABELS
from model.model import ToolSegmenterModel
from helpers.image import overlay_segmentation


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
            self._test_save_dir = self.config['testing_opts']['save_dir']
        else:
            self._test_minibatch_size = 16
            self._test_final_image_size_hw = [1080, 1920]
            self._test_save_dir = "test_results"

    def _targets_to_instances(self, mask: torch.Tensor) -> Instances:
        """
        Given a segmentation mask, create binary masks for each class and store it into an Instances object.
        Required for compatibility with Mask2Former loss
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
            processed_results = self._postprocess_outputs(outputs, (imgs.shape[-2], imgs.shape[-1]), "bilinear", imgs.device)
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


    def test_step(self, batch, batch_idx):
        """
        Runs prediction on a list of ordered frames of a video.
        Args:
            batch: a dict containing the following keys:
                "video_name": a string containing the name of the video
                "images": a tensor of shape (N, C, H, W) containing the images
                "masks": a tensor of shape (N, 1, H, W) containing the masks
        """
        imgs = batch['images'][0] # PL lightning returns a 5D tensor here
        num_batches = imgs.shape[0] // self._test_minibatch_size
        if imgs.shape[0] % self._test_minibatch_size != 0:
            num_batches += 1

        outputs = []
        # Prevent OOMs on long sequences
        for i in range(num_batches):
            imgs_batch = imgs[i * self._test_minibatch_size:(i + 1) * self._test_minibatch_size]
            b_out = self.model(imgs_batch)
            outputs.append(b_out)

        # Merge output dicts
        merged_outputs = {}
        for ok in ['pred_logits', 'pred_masks']:
            merged_outputs[ok] = torch.cat([o[ok] for o in outputs], dim=0).to('cpu')

        # Free GPU memory
        del outputs

        resize_sz = [self._test_final_image_size_hw[0], self._test_final_image_size_hw[1]]
        processed_results = self._postprocess_outputs(merged_outputs, resize_sz, "bilinear", "cpu")
        processed_results = torch.stack([result.argmax(dim=0) for result in processed_results])

        preds_dir = self._test_save_dir + "/predictions"
        mask_names = [y for x in batch['mask_names'] for y in x]
        video_name = basename(dirname(dirname(mask_names[0])))

        case_preds_dir = os.path.join(preds_dir, video_name, 'segmentation')
        os.makedirs(case_preds_dir, exist_ok=True)
        # Do this in a thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self._save_video_predictions, mask_names, processed_results, case_preds_dir)


    def _save_video_predictions(self, mask_names: List[str], preds: torch.Tensor, case_preds_dir: str):
        preds_np = preds.cpu().numpy()
        for j in range(preds_np.shape[0]):
            pred_upsampled_i = np.repeat(preds_np[j][None], 3, axis=0)
            pred_upsampled_i = pred_upsampled_i.transpose(1, 2, 0).astype(np.uint8)
            Image.fromarray(pred_upsampled_i).save(os.path.join(case_preds_dir, basename(mask_names[j])))

        print(f"Finished saving video predictions in dir {case_preds_dir}")


    @torch.no_grad()
    def _postprocess_outputs(self, outputs: Dict[str, torch.Tensor], resize_sz: Tuple[int, int], interp_mode: str, interp_device: str) -> torch.Tensor:
        """
        Args:
            outputs: a dict containing the following keys:
                "pred_logits": a tensor of shape (B, Q, K+1)
                "pred_masks": a tensor of shape (B, Q, H, W)
                With Q the number of queries and K+1 the number of classes + no object class
            resize_sz: Tuple (H, W) to resize to
            interp_mode: interpolation mode
        Returns:
            a tensor of shape (B, K, H, W) containing the logits of the segmentation masks
        """
        mask_cls_results = outputs["pred_logits"].to(interp_device)
        mask_pred_results = outputs["pred_masks"].to(interp_device)

        upsampled_results = []
        align_corners = False if interp_mode in ['bilinear', 'bicubic'] else None

        upsampled_results = F.interpolate(
            mask_pred_results,
            size=resize_sz,
            mode=interp_mode,
            align_corners=align_corners,
        )

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, upsampled_results
        ):

            # on GPU for einsum
            r = self.model.semantic_inference(mask_cls_result.to('cuda'), mask_pred_result.to('cuda'))
            processed_results.append(r.to(interp_device))

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
            else:
                raise ValueError(f"Scheduler {self.config['training_opts']['scheduler']} not supported")
            result_dict['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            }

        return result_dict