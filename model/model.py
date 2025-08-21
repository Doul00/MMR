
import yaml
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from model.sam2.sam.mask_decoder import MaskDecoder


class ToolSegmenterModel(nn.Module):
    def __init__(self,
                 backbone_name: str = 'hiera_tiny_224',
                 backbone_ckpt: str = 'mae_in1k'):
        super(ToolSegmenterModel, self).__init__()
        self.backbone = torch.hub.load("facebookresearch/hiera",
                                       model=backbone_name,
                                       checkpoint=backbone_ckpt,
                                       pretrained=True)
        self.decoder = None # FIXME

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)