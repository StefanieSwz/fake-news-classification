"""This script downloads the dataset from Kaggle and unzips it."""

import os
import subprocess
import zipfile

from fakenews.config import DATASET, RAW_DATA_DIR

def download_dataset(dataset_name, save_dir):
    """
    Function to download dataset from Kaggle.
    Args:
        dataset_name (str): Name of the dataset on Kaggle.
        save_dir (str): Directory to save the downloaded dataset.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Check if dataset is already present locally
    dataset_zip_path = os.path.join(save_dir, f"{dataset_name.split('/')[1]}.zip")
    if os.path.exists(dataset_zip_path):
        print(f"{dataset_zip_path}: Skipping, found more recently modified local copy")
        return

    # Download dataset if not present locally
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_name, "-p", save_dir]
    )


def unzip_dataset(zip_file_path, extract_dir):
    """
    Function to unzip dataset.
    Args:
        zip_file_path (str): Path to the zip file.
        extract_dir (str): Directory to extract the zip file.
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


if __name__ == "__main__":
    dataset_name = DATASET
    save_dir = RAW_DATA_DIR

    download_dataset(dataset_name, save_dir)

    zip_file_path = os.path.join(save_dir, f"{dataset_name.split('/')[1]}.zip")
    extract_dir = save_dir

    unzip_dataset(zip_file_path, extract_dir)
