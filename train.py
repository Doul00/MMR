"""
@brief: Main training script
"""

import hydra
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from model.module import ToolSegmenterModule
from data.datasets import SARRARP50DataModule


@hydra.main(config_path="conf", config_name="train_cfg", version_base=None)
def main(cfg: DictConfig):
    module = ToolSegmenterModule(config=cfg)
    data_module = SARRARP50DataModule(config=cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{cfg.log_dir}/{timestamp}"

    logger = TensorBoardLogger(save_dir=log_dir, name=cfg.exp_name)
    model_ckpt_callback = ModelCheckpoint(dirpath=log_dir + "/weights",
                      monitor='val_loss',
                      mode='min',
                      save_top_k=1,
                      save_last=True)
    trainer = Trainer(max_epochs=cfg['training_opts']['max_epochs'],
                      logger=logger,
                      precision="16-mixed",
                      callbacks=[model_ckpt_callback])

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()