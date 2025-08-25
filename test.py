"""
@brief: Script used to generate model predictions
"""

import os
import hydra
import random
import numpy as np
import torch
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from omegaconf import DictConfig
from model.module import ToolSegmenterModule
from data.datasets import SARRARP50DataModule


@hydra.main(config_path="conf", config_name="train_cfg", version_base=None)
def main(cfg: DictConfig):

    random.seed(cfg['training_opts']['seed'])
    np.random.seed(cfg['training_opts']['seed'])
    torch.manual_seed(cfg['training_opts']['seed'])

    ckpt_path = cfg.ckpt_path
    cfg_file = os.path.dirname(ckpt_path) + "/../train_cfg.yaml"
    train_cfg = OmegaConf.load(cfg_file)
    train_cfg['data_opts']['test_dataset']['listfile'] = cfg.data_opts.test_dataset.listfile
    train_cfg['testing_opts']['save_dir'] = cfg.testing_opts.save_dir
    print("Your results will be saved in: ", cfg.testing_opts.save_dir)

    module = ToolSegmenterModule.load_from_checkpoint(ckpt_path, config=train_cfg)
    data_module = SARRARP50DataModule(config=train_cfg)

    module.eval()

    trainer = Trainer(max_epochs=cfg['training_opts']['max_epochs'], precision="16")
    trainer.test(module, datamodule=data_module)


if __name__ == "__main__":
    main()