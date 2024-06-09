"""Tests for the data module."""

import os
import pytest
from unittest.mock import patch
import pandas as pd
import tempfile
import shutil
import zipfile
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.data.make_dataset import download_dataset, unzip_dataset

@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

@pytest.fixture
def data_preprocessor(temp_dir):
    preprocessor = DataPreprocessor(temp_dir, max_length=128)
    data = pd.DataFrame({
        "title": ["Fake news title 1", "Fake news title 2"],
        "label": [1, 0]
    })
    data.to_csv(os.path.join(temp_dir, "WELFake_Dataset.csv"), index=False)
    return preprocessor

def test_load_data(data_preprocessor):
    data = data_preprocessor.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data(data_preprocessor):
    data = pd.DataFrame({
        "title": ["Fake news title 1", "Fake news title 2"],
        "label": [1, 0]
    })
    preprocessed_data = data_preprocessor.preprocess_data(data)
    assert "label" in preprocessed_data.columns
    assert pd.api.types.is_integer_dtype(preprocessed_data["label"])

def test_save_preprocessed_data(data_preprocessor, temp_dir):
    data = pd.DataFrame({
        "title": ["Fake news title 1", "Fake news title 2"],
        "label": [1, 0]
    })
    preprocessed_data = data_preprocessor.preprocess_data(data)
    data_preprocessor.save_preprocessed_data(preprocessed_data, temp_dir)
    saved_file_path = os.path.join(temp_dir, "preprocessed_data.csv")
    assert os.path.exists(saved_file_path)

def test_load_preprocessed_data(data_preprocessor, temp_dir):
    data = pd.DataFrame({
        "title": ["Fake news title 1", "Fake news title 2"],
        "label": [1, 0]
    })
    preprocessed_data = data_preprocessor.preprocess_data(data)
    data_preprocessor.save_preprocessed_data(preprocessed_data, temp_dir)
    loaded_data = data_preprocessor.load_preprocessed_data(temp_dir)
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty

@patch('subprocess.run')
def test_download_dataset(mock_run, temp_dir):
    dataset_name = "someuser/somedataset"
    download_dataset(dataset_name, temp_dir)
    dataset_zip_path = os.path.join(temp_dir, f"{dataset_name.split('/')[1]}.zip")
    mock_run.assert_called_with(["kaggle", "datasets", "download", "-d", dataset_name, "-p", temp_dir])
    assert os.path.exists(temp_dir)

@patch('zipfile.ZipFile.extractall')
def test_unzip_dataset(mock_extractall, temp_dir):
    # Create a valid ZIP file
    zip_file_path = os.path.join(temp_dir, "test.zip")
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Create a file to include in the ZIP archive
        test_txt_path = os.path.join(temp_dir, "test.txt")
        with open(test_txt_path, 'w') as f:
            f.write("Test content")
        zipf.write(test_txt_path, arcname="test.txt")
    
    # Test the unzip functionality
    unzip_dataset(zip_file_path, temp_dir)
    mock_extractall.assert_called_once()
