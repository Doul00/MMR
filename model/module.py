"""
@brief: ToolSegmenterModule pytorch lightning module. Handles training loop and inference
"""

import torch
import numpy as np
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch.nn import functional as F
from monai.losses import GeneralizedDiceLoss
from detectron2.structures import Instances, ImageList
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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

    def forward(self, x):
        return self.model(x)

    def _targets_to_instances(self, mask):
        instances = Instances(image_size=(mask.shape[-2], mask.shape[-1]))
        instances.gt_classes = torch.LongTensor(list(SEG_LABELS.values())).to(mask.device)
        bit_mask_tensor = torch.zeros((len(instances.gt_classes), mask.shape[-2], mask.shape[-1])).to(mask.device)
        for val in instances.gt_classes:
            bit_mask = torch.zeros_like(mask)
            bit_mask[mask == val] = 1
            bit_mask_tensor[val] = bit_mask

            # # Safety check
            # from PIL import Image
            # import numpy as np
            # bit_mask_np = (255 * bit_mask.squeeze().cpu().numpy()).astype(np.uint8)
            # Image.fromarray(bit_mask_np).save(f"bit_mask_{val}.png")

        # from matplotlib import cm
        # num_classes = len(SEG_LABELS)
        # cmap = cm.get_cmap('tab10', num_classes)
        # mask_np = mask.squeeze().cpu().numpy().astype(np.float32)
        # rgba_img = cmap(mask_np / (num_classes - 1))
        # # Transparent background
        # rgba_img[mask_np == 0, -1] = 0
        # rgba_img = (255 * rgba_img).astype(np.uint8)
        # Image.fromarray(rgba_img).save(f"mask.png")

        instances.gt_masks = bit_mask_tensor
        return instances

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        outputs = self.model(imgs)
        targets = [self._targets_to_instances(target) for target in targets]
        targets = self.model.prepare_targets(targets, imgs)

        # bipartite matching-based loss
        losses = self.model.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.model.criterion.weight_dict:
                losses[k] *= self.model.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        total_loss = torch.stack(list(losses.values())).sum()
        self.log('train_loss', total_loss, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        outputs = self.model(imgs)
        targets = [self._targets_to_instances(mask) for mask in masks]
        targets = self.model.prepare_targets(targets, imgs)

        # bipartite matching-based loss
        losses = self.model.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.model.criterion.weight_dict:
                losses[k] *= self.model.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        total_loss = torch.stack(list(losses.values())).sum()
        self.log('val_loss', total_loss, prog_bar=True, logger=True)

        # Visualize results
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

    def _postprocess_outputs(self, imgs, outputs):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(imgs.shape[-2], imgs.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):

            r = self.model.semantic_inference(mask_cls_result, mask_pred_result)
            processed_results.append(r)

        return processed_results


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        result_dict = {'optimizer': optimizer}
        if 'scheduler' in self.config['training_opts']:
            if self.config['training_opts']['scheduler'] == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=self.config['training_opts']['max_epochs'], eta_min=1e-5)
            elif self.config['training_opts']['scheduler'] == 'plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
            else:
                raise ValueError(f"Scheduler {self.config['training_opts']['scheduler']} not supported")
            result_dict['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }

        return result_dict