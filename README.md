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
- Data is expected to be in the same format as described in https://github.com/surgical-vision/SAR_RARP50-evaluation, i.e:

```
path_to_root_data_dir
├── train1
│   ├── video_*
│   ├── ...
│   └── video_*
├── train2
│   ├── video_*
│   ├── ...
│   └── video_*
└── test
    ├── video_*
    ├── ...
    └── video_*
```

## Training

### Mask2Former model
- Firstly, generate files listing the paths to the segmentation labels used for training, similarly to the files in `data/splits_final/*.txt`
- A util script is provided in `scripts/data/make_splits.sh` to split the training set.
- Call `scripts/train/model_train.sh` and adjust the paths to your listfiles.

### nnUNet model
- First, generate data in the format expected by nnUnet using `scripts/data/generate_nnunet_dataset.sh`
- Then call `scripts/data/nnunet_plan_preprocess.sh` to prepare the dataset
- Finally use `scripts/train/nnunet_train.sh` then `scripts/train/nnunet_best_cfg.sh` to find the best nnUnet configuration
- Save the commands printed by the latest script to generate the predictions

## Generating predictions

### Mask2Former model
- Use `scripts/test/model_test.sh` and update the path to the checkpoint.

### nnUnet
- Used the commands printed by `nnunet_best_cfg.sh`. An example script is provided in `scripts/test/nnunet_eval.sh`

## Evaluating
- Clone the SAR_RARPO50 challenge evaluation repo `git clone https://github.com/surgical-vision/SAR_RARP50-evaluation`
- Call `scripts/test/evaluate_model_results.sh` and update the path to your model predictions and your data directory with the labels. 