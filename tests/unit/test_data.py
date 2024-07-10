"""
Tests for the data module.
"""

import os
import shutil
import tempfile
from unittest.mock import patch, mock_open
import zipfile
import pandas as pd
import pytest
import torch

from fakenews.data.make_dataset import download_dataset, unzip_dataset
from fakenews.data.preprocessing import DataPreprocessor

from fakenews.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


@pytest.fixture
def temp_dir():
    """Fixture to create and clean up a temporary directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def data_preprocessor(temp_dir):
    """Fixture to create a DataPreprocessor with sample data."""
    preprocessor = DataPreprocessor(temp_dir, max_length=128)
    data = pd.DataFrame(
        {
            "title": ["Fake news title 1", "Fake news title 2"],
            "text": ["Fake news text 1", "Fake news text 2"],
            "label": [1, 0],
        }
    )
    data.to_csv(os.path.join(temp_dir, "WELFake_Dataset.csv"), index=False)
    return preprocessor


def test_split_data(data_preprocessor):
    """
    Test the split_data function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    data = pd.DataFrame(
        {
            "title": [
                "Fake news title 1",
                "Fake news title 2",
                "Fake news title 3",
                "Fake news title 4",
                "Fake news title 5",
                "Fake news title 6",
                "Fake news title 7",
                "Fake news title 8",
                "Fake news title 9",
                "Fake news title 10",
                "Fake news title 11",
                "Fake news title 12",
                "Fake news title 13",
                "Fake news title 14",
                "Fake news title 15",
                "Fake news title 16",
            ],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    train_text, val_text, test_text, train_labels, val_labels, test_labels = data_preprocessor.split_data(
        data, test_size=0.25, val_size=0.5, random_state=42
    )
    assert len(train_text) == 12
    assert len(val_text) == 2
    assert len(test_text) == 2
    assert len(train_labels) == 12
    assert len(val_labels) == 2
    assert len(test_labels) == 2


def test_tokenize_data(data_preprocessor):
    """
    Test the tokenize_data function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    texts = [
        "Fake news title 1",
        "Fake news title 2",
        "Fake news title 3",
        "Fake news title 4",
        "Fake news title 5",
        "Fake news title 6",
        "Fake news title 7",
        "Fake news title 8",
        "Fake news title 9",
        "Fake news title 10",
        "Fake news title 11",
        "Fake news title 12",
        "Fake news title 13",
        "Fake news title 14",
        "Fake news title 15",
        "Fake news title 16",
    ]

    tokens = data_preprocessor.tokenize_data(texts)
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert len(tokens["input_ids"]) == 16
    assert len(tokens["attention_mask"]) == 16


def test_create_tensors(data_preprocessor):
    """
    Test the create_tensors function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    tokens = {
        "input_ids": [[101, 2054, 2003, 1996, 2568, 102], [101, 2129, 2079, 2017, 4067, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    }
    labels = [1, 0]
    seq, mask, y = data_preprocessor.create_tensors(tokens, labels)
    assert isinstance(seq, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert seq.shape == (2, 6)
    assert mask.shape == (2, 6)
    assert y.shape == (2,)


def test_create_dataloaders(data_preprocessor):
    """
    Test the create_dataloaders function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    train_seq = torch.tensor([[101, 2054, 2003, 1996, 2568, 102], [101, 2129, 2079, 2017, 4067, 102]])
    train_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    train_y = torch.tensor([1, 0])
    val_seq = train_seq
    val_mask = train_mask
    val_y = train_y
    test_seq = train_seq
    test_mask = train_mask
    test_y = train_y

    train_dataloader, val_dataloader, test_dataloader = data_preprocessor.create_dataloaders(
        train_seq,
        train_mask,
        train_y,
        val_seq,
        val_mask,
        val_y,
        test_seq,
        test_mask,
        test_y,
        batch_size=2,
    )
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)
    assert isinstance(test_dataloader, torch.utils.data.DataLoader)


def test_preprocess_csv(data_preprocessor):
    """
    Test the preprocess_csv function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    with patch("builtins.open", mock_open(read_data="title\nFake news title 1\nFake news title 2\n")):
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"title": ["Fake news title 1", "Fake news title 2"]})
            data = data_preprocessor.preprocess_csv("/fake/csv_path.csv")
            assert not data.empty
            assert "title" in data.columns


# Tests using fixtures


def test_load_data(data_preprocessor):
    """
    Test the load_data function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    data = data_preprocessor.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_preprocess_data(data_preprocessor):
    """
    Test the preprocess_data function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
    """
    data = data_preprocessor.load_data()
    preprocessed_data = data_preprocessor.preprocess_data(data)
    assert pd.api.types.is_integer_dtype(preprocessed_data["label"])


def test_save_preprocessed_data(data_preprocessor, temp_dir):
    """
    Test the save_preprocessed_data function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
        temp_dir: Fixture providing a temporary directory.
    """
    data = pd.DataFrame(
        {
            "title": ["Fake news title 1", "Fake news title 2"],
            "text": ["Fake news text 1", "Fake news text 2"],
            "label": [1, 0],
        }
    )
    preprocessed_data = data_preprocessor.preprocess_data(data)
    data_preprocessor.save_preprocessed_data(preprocessed_data, temp_dir)
    saved_file_path = os.path.join(temp_dir, "preprocessed_data.csv")
    assert os.path.exists(saved_file_path)


def test_load_preprocessed_data(data_preprocessor, temp_dir):
    """
    Test the load_preprocessed_data function of DataPreprocessor.

    Args:
        data_preprocessor: Fixture providing a DataPreprocessor instance.
        temp_dir: Fixture providing a temporary directory.
    """
    preprocessed_data = pd.DataFrame(
        {
            "title": ["Fake news title 1", "Fake news title 2"],
            "label": [1, 0],
        }
    )
    data_preprocessor.save_preprocessed_data(preprocessed_data, temp_dir)
    loaded_data = data_preprocessor.load_preprocessed_data(temp_dir, "preprocessed_data.csv")
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty


@patch("subprocess.run")
def test_download_dataset(mock_run, temp_dir):
    """
    Test the download_dataset function.

    Args:
        mock_run: Mock object for subprocess.run.
        temp_dir: Fixture providing a temporary directory.
    """
    dataset_name = "saurabhshahane/fake-news-classification"
    download_dataset(dataset_name, temp_dir)
    mock_run.assert_called_with(["kaggle", "datasets", "download", "-d", dataset_name, "-p", temp_dir])
    assert os.path.exists(temp_dir)


@patch("zipfile.ZipFile.extractall")
def test_unzip_dataset(mock_extractall, temp_dir):
    """
    Test the unzip_dataset function.

    Args:
        mock_extractall: Mock object for ZipFile.extractall.
        temp_dir: Fixture providing a temporary directory.
    """
    zip_file_path = os.path.join(temp_dir, "test.zip")
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        test_txt_path = os.path.join(temp_dir, "test.txt")
        with open(test_txt_path, "w") as f:
            f.write("Test content")
        zipf.write(test_txt_path, arcname="test.txt")

    unzip_dataset(zip_file_path, temp_dir)
    mock_extractall.assert_called_once()


# Tests for actual data


@pytest.fixture
def raw_data():
    """Fixture to provide the raw dataset."""
    file_path = os.path.join(RAW_DATA_DIR, "WELFake_Dataset.csv")
    df = pd.read_csv(file_path)
    return df


@pytest.fixture
def preprocessed_data():
    """Fixture to provide the preprocessed dataset."""
    file_path = os.path.join(PROCESSED_DATA_DIR, "preprocessed_data.csv")
    df = pd.read_csv(file_path)
    return df


def test_not_empty(raw_data):
    """
    Test if the raw dataset is not empty.

    Args:
        raw_data: Fixture providing the raw dataset.
    """
    assert not raw_data.empty, "The dataset is empty"


def test_columns_exist(raw_data):
    """
    Test if the expected columns exist in the raw dataset.

    Args:
        raw_data: Fixture providing the raw dataset.
    """
    expected_columns = ["title", "text", "label"]  # Add any other expected columns here
    for column in expected_columns:
        assert column in raw_data.columns, f"Missing column: {column}"


def test_title_is_string(preprocessed_data):
    """
    Test if the 'title' column in the preprocessed dataset contains only strings.

    Args:
        preprocessed_data: Fixture providing the preprocessed dataset.
    """
    assert (
        preprocessed_data["title"].apply(lambda x: isinstance(x, str)).all()
    ), "Title column contains non-string values"


def test_data_shape():
    """
    Test the shape of the dataset loaded by DataPreprocessor.
    """
    data_preprocessor = DataPreprocessor(RAW_DATA_DIR, max_length=128)
    data = data_preprocessor.load_data()
    assert data.shape[0] == 72134
    assert "title" in data.columns
    assert "label" in data.columns
    assert "text" in data.columns


def test_preprocess_real_data():
    """
    Test the preprocessing of the real dataset by DataPreprocessor.
    """
    data_preprocessor = DataPreprocessor(RAW_DATA_DIR, max_length=128)
    data = data_preprocessor.load_data()
    data = data_preprocessor.preprocess_data(data)
    assert pd.api.types.is_integer_dtype(data["label"])
