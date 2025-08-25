# SAR_RARP50 segmentation challenge

![Teaser](teaser.gif)

Repo architecture:
- conf: Configuration files
- data: Data related scripts (processing / data loading...) & listfiles for training / testing
- model: Model architecture
- helpers: Utils functions
- scripts: Scripts for training / testing and data processing
- results: CSV files with results per fold
- weights: Model weights (Not pushed to the repo but will be shared separately)

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
- An util script is provided in `scripts/data/make_splits.sh` to split the training set.
- Call `scripts/train/model_train.sh` and adjust the paths to your listfiles.

### nnUNet model
- First, generate data in the format expected by nnUnet using `scripts/data/generate_nnunet_dataset.sh`
- Then call `scripts/data/nnunet_plan_preprocess.sh` to prepare the dataset
- Finally use `scripts/train/nnunet_train.sh` then `scripts/train/nnunet_best_cfg.sh` to find the best nnUnet configuration
- Save the commands printed by the latest script to generate the predictions

## Generating predictions

### Mask2Former model
- Use `scripts/test/model_test.sh` and update the path to the checkpoint.
- Merge predictions using  `helpers/merge_predictions.py`

### nnUnet
- Used the commands printed by `nnunet_best_cfg.sh`. An example script is provided in `scripts/test/nnunet_eval.sh`
- If using provided weights, use `nnUNetv2_install_pretrained_model_from_zip PATH_TO_WEIGHTS_FOLDER/nnunet/nnunet_best.zip`. You can then run the eval script.

## Evaluating
- Clone the SAR_RARP50 challenge evaluation repo `git clone https://github.com/surgical-vision/SAR_RARP50-evaluation`
- Call `scripts/test/evaluate_model_results.sh` and update the path to your model predictions and your data directory with the labels. 


## Results

| method             | metric | video_41 | video_42 | video_43 | video_44 | video_45 | video_46 | video_47 | video_48 | video_49 | video_50 |  mean  |
|--------------------|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|--------|
| M2F (ensemble)     | mIoU   |   0.807  |   0.760  |   0.767  |   0.734  |  0.779   |  0.764   |  0.781   |  0.830   |  0.709   |  0.766   | 0.770  |
|                    | mNSD   |   0.825  |   0.783  |   0.785  |   0.768  |  0.837   |  0.808   |  0.830   |  0.860   |  0.747   |  0.836   | 0.810  |
| nnUNet (ensemble)  | mIoU   |   0.845  |   0.767  |   0.802  |   0.741  |  0.808   |  0.802   |  0.787   |  0.859   |  0.758   |  0.823   | 0.800  |
|                    | mNSD   |   0.870  |   0.795  |   0.829  |   0.775  |  0.861   |  0.842   |  0.836   |  0.887   |  0.801   |  0.892   | 0.840  |
