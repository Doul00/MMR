"""
@brief: Torch model for tool segmentation.
"""
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone, build_sem_seg_head

from model.modeling.criterion import SetCriterion
from model.modeling.matcher import HungarianMatcher



class ToolSegmenterModel(nn.Module):
    def __init__(self, cfg):
        """
        ToolSegmentation model with a SwinUNet backbone and a MaskFormer head.
        """
        super().__init__()
        self.backbone = build_backbone(cfg, input_shape=ShapeSpec(channels=3))
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            self.sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.criterion = criterion
        self.overlap_threshold = cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD
        self.object_mask_threshold = cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD

    def forward(self, img):
        """
        Args:
            img: a tensor of shape (B, C, H, W)
        """
        features = self.backbone(img)
        outputs = self.sem_seg_head(features)
        return outputs

    def prepare_targets(self, targets, images):
        """Optional padding of the targets to the input image size."""
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        """
        Performs a weighted sum of the confidence scores from the queries with the predicted masks.
        With K the number of classes and Q the number of queries:
        Args:
            mask_cls: a tensor of shape (Q, K+1) containing the logits of the classification scores
            mask_pred: a tensor of shape (Q, H, W) containing the logits of the segmentation masks
        Returns:
            a tensor of shape (K, H, W) containing the logits of the segmentation masks
        """
        # Remove no object class
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
