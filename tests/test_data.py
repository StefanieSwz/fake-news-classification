"""Tests for the data module."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch
import pandas as pd
import tempfile
import shutil
import zipfile
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.data.make_dataset import download_dataset, unzip_dataset
from fakenews.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

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
        "text": ["Fake news text 1", "Fake news text 2"],
        "label": [1, 0]
    })
    data.to_csv(os.path.join(temp_dir, "WELFake_Dataset.csv"), index=False)
    return preprocessor

"""
def test_load_data():
    data_preprocessor = DataPreprocessor(RAW_DATA_DIR, max_length=128)
    data = data_preprocessor.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_data_shape():
    data_preprocessor = DataPreprocessor(RAW_DATA_DIR, max_length=128)
    data = data_preprocessor.load_data()
    assert data.shape[0] == 72134
    assert "title" in data.columns
    assert "label" in data.columns
    assert "text" in data.columns

def test_preprocess_data(data_preprocessor):
    data_preprocessor = DataPreprocessor(RAW_DATA_DIR, max_length=128)
    data = data_preprocessor.load_data()
    data = data_preprocessor.preprocess_data(data)
    assert pd.api.types.is_integer_dtype(data["label"])
"""
def test_save_preprocessed_data(data_preprocessor, temp_dir):
    data = pd.DataFrame({
        "title": ["Fake news title 1", "Fake news title 2"],
        "text": ["Fake news text 1", "Fake news text 2"],
        "label": [1, 0]
    })
    preprocessed_data = data_preprocessor.preprocess_data(data)
    data_preprocessor.save_preprocessed_data(preprocessed_data, temp_dir)
    saved_file_path = os.path.join(temp_dir, "preprocessed_data.csv")
    assert os.path.exists(saved_file_path)
"""
def test_load_preprocessed_data():
    data_preprocessor = DataPreprocessor(RAW_DATA_DIR, max_length=128)
    loaded_data = data_preprocessor.load_preprocessed_data(PROCESSED_DATA_DIR)
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty
"""
@patch('subprocess.run')
# @patch temporarily replaces the specified object with a Mock object; after the test, the original object is restored
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
