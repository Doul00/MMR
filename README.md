# SAR_RARP50 segmentation challenge

Repo architecture:
- conf: Configuration files
- data: Data related scripts (processing / data loading...) & listfiles for training / testing
- model: Model architecture
- helpers: Utils functions
- scripts: Scripts for training / testing and data processing
- weights: Model weights (Not pushed to the repo but included in the zipfile)

## Installation
- Run `pip install -r requirements.txt` to install dependencies

## Training
- Call `scripts/train/model
- Clone the SAR_RARPO50 challenge evaluation repo and cd into it: `git clone https://github.com/surgical-vision/SAR_RARP50-evaluation`