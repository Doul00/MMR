"""
@brief: ToolSegmenterModule pytorch lightning module. Handles training loop and inference
"""

import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch.nn import functional as F
from monai.losses import GeneralizedDiceLoss

from model.model import ToolSegmenterModel
from helpers.image import overlay_segmentation


class ToolSegmenterModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = ToolSegmenterModel(self.config['model_opts'])
        training_opts = self.config['training_opts']
        self.lr = training_opts['learning_rate']
        self.weight_decay = training_opts['weight_decay']
        self.optimizer = training_opts['optimizer']
        self.scheduler = training_opts['scheduler']
        self.loss_fn = GeneralizedDiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
