"""
@brief: Main training script
"""

import argparse
from pytorch_lightning import Trainer

from model.module import ToolSegmenterModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a tool segmenter model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    module = ToolSegmenterModule(config_path=args.config)

    trainer = Trainer(max_epochs=module.config['training_opts']['max_epochs'])

    # Start training
    trainer.fit(module)