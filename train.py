"""
@brief: Main training script
"""

import os
import hydra
import random
import numpy as np
import torch
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from model.module import ToolSegmenterModule
from data.datasets import SARRARP50DataModule


@hydra.main(config_path="conf", config_name="train_cfg", version_base=None)
def main(cfg: DictConfig):

    random.seed(cfg['training_opts']['seed'])
    np.random.seed(cfg['training_opts']['seed'])
    torch.manual_seed(cfg['training_opts']['seed'])

    module = ToolSegmenterModule(config=cfg)
    data_module = SARRARP50DataModule(config=cfg)

    log_dir = cfg.log_dir + "/" + cfg.exp_name
    os.makedirs(log_dir, exist_ok=True)

    with open(log_dir + "/train_cfg.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = TensorBoardLogger(save_dir=log_dir, name='tb_logs')
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    model_ckpt_callback = ModelCheckpoint(dirpath=log_dir + "/weights",
                      filename='{epoch}-{val_loss:.3f}',
                      monitor='val_loss',
                      mode='min',
                      save_top_k=1,
                      save_last=True)

    num_nodes = int(os.getenv("NUM_NODES", 1))
    trainer = Trainer(max_epochs=cfg['training_opts']['max_epochs'],
                      logger=logger,
                      precision="16",
                      devices="auto",
                      num_nodes=num_nodes,
                      log_every_n_steps=5,
                      callbacks=[model_ckpt_callback, lr_callback])

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()