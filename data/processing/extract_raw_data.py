"""
@brief: Extracts all zip files containing the video challenge data
"""

import os
import sys
import zipfile
import multiprocessing as mp

from os.path import dirname, basename

def extract_zip(zip_path, out_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

if __name__ == '__main__':
    data_dir = sys.argv[1]
    zip_files = [os.path.join(root, file)
                 for root, _, files in os.walk(data_dir)
                 for file in files if file.endswith('.zip')]

    proc_args = [(zf, dirname(zf) + "/" + basename(zf).replace('.zip', '')) for zf in zip_files]
    with mp.Pool(8) as pool:
        pool.starmap(extract_zip, proc_args)
