
import yaml
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from omegaconf import DictConfig

from model.modeling.backbone import Backbone, build_backbone
from model.modeling.meta_arch import build_sem_seg_head
from model.modeling.postprocessing import sem_seg_postprocess
from model.modeling.structures import Boxes, ImageList, Instances, BitMasks

from model.modeling.criterion import SetCriterion
from model.modeling.matcher import HungarianMatcher



class ToolSegmenterModel(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
        """
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MASK_FORMER.DEC_LAYERS
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
            num_points=cfg.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        self.num_queries = cfg.MASK_FORMER.NUM_OBJECT_QUERIES
        self.criterion = criterion
        self.size_divisibility = cfg.MASK_FORMER.SIZE_DIVISIBILITY
        if self.size_divisibility < 0:
            # use backbone size_divisibility if not set
            self.size_divisibility = self.backbone.size_divisibility
        self.overlap_threshold = cfg.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
        self.object_mask_threshold = cfg.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,

    def forward(self, img):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        features = self.backbone(img)
        outputs = self.sem_seg_head(features)
        return outputs

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
        else:
            targets = None

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        images, batched_inputs = batch
        outputs = self.forward(batched_inputs)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            r = self.semantic_inference(mask_cls_result, mask_pred_result)
            processed_results[-1]["sem_seg"] = r

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
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
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
