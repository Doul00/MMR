@echo OFF

SET "CURR_DIR=%~dp0"
SET ROOT_DIR=%CURR_DIR%..\..
set PYTHONPATH=%ROOT_DIR%

SET nnUNet_raw=E:/nnunet_raw
SET nnUNet_preprocessed=E:/nnunet_preprocessed
SET nnUNet_results=E:/nnunet_results

python %ROOT_DIR%\data\generate_nnunet_dataset.py %*
