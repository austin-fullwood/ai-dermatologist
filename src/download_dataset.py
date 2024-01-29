#! python

import os
from pathlib import Path
import shutil

import opendatasets as od

"""
Downloads the Skin Cancer MNIST: HAM10000 dataset from Kaggle. '
"""
def download_dataset() -> None:
    directory = Path(__file__).resolve().parent
    if os.path.exists(directory / "skin-cancer-mnist-ham10000"):
        print("Dataset already downloaded. Redownload? (y/n)")
        if input() != "y":
            exit()
        shutil.rmtree(directory / "skin-cancer-mnist-ham10000")


    print("--- Downloading dataset ---")

    url = "https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000"
    od.download(url) # TODO: Remove 3rd party dependency

    print("--- Dataset downloaded ---")

if __name__ == "__main__":
    download_dataset()
