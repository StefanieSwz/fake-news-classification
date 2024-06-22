import os

import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizerFast

from fakenews.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


class DataPreprocessor:
    """
    A class to preprocess data for BERT model training.

    Attributes:
        data_dir (str): The directory where the data files are stored.
        max_length (int): The maximum length of the tokenized sequences.
        tokenizer (BertTokenizerFast): The tokenizer for BERT model.
    """

    def __init__(self, data_dir, max_length):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Args:
            data_dir (str): The directory where the data files are stored.
            max_length (int): The maximum length of the tokenized sequences.
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def load_data(self):
        """
        Loads the WELFake dataset.

        Returns:
            pd.DataFrame: DataFrame containing the dataset.
        """
        data = pd.read_csv(os.path.join(self.data_dir, "WELFake_Dataset.csv"))
        return data

    def preprocess_data(self, data):
        """
        Preprocesses the data by generating labels and keeping indices.

        Args:
            data (pd.DataFrame): DataFrame containing the dataset.

        Returns:
            pd.DataFrame: DataFrame containing the preprocessed data with labels and indices.
        """
        data = data.dropna(subset=["title"])  # Drop rows where title is NaN
        data = data.reset_index(drop=True)  # Reset index to get row indices without column name
        data["label"] = data["label"].astype(int)
        return data[["title", "label"]]

    def save_preprocessed_data(self, data, processed_data_dir):
        """
        Saves the preprocessed data to a CSV file.

        Args:
            data (pd.DataFrame): DataFrame containing the preprocessed data.
            processed_data_dir (str): Directory to save the preprocessed data.
        """
        preprocessed_file = os.path.join(processed_data_dir, "preprocessed_data.csv")
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        data.to_csv(preprocessed_file, index=True)  # Save the DataFrame with the index
        print(f"Preprocessed data saved to {preprocessed_file}")

    def load_preprocessed_data(self, processed_data_dir, file_name):
        """
        Loads the preprocessed data from a CSV file.

        Args:
            processed_data_dir (str): Directory to load the preprocessed data from.

        Returns:
            pd.DataFrame: DataFrame containing the preprocessed data.
        """
        preprocessed_file = os.path.join(processed_data_dir, file_name)
        data = pd.read_csv(preprocessed_file, index_col=0)  # Load the DataFrame with the index
        return data

    def split_data(self, data, test_size, val_size, random_state):
        """
        Splits the data into training, validation, and test sets.

        Args:
            data (pd.DataFrame): The preprocessed data.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the test dataset to include in the validation split.
            random_state (int): Random seed for shuffling the data.

        Returns:
            tuple: Containing lists of training text data, validation text data, test text data,
            training labels, validation labels, and test labels.
        """
        train_text, temp_text, train_labels, temp_labels = train_test_split(
            data["title"],
            data["label"],
            random_state=random_state,
            test_size=test_size,
            stratify=data["label"],
        )
        val_text, test_text, val_labels, test_labels = train_test_split(
            temp_text,
            temp_labels,
            random_state=random_state,
            test_size=val_size,
            stratify=temp_labels,
        )
        train_text = train_text.tolist()
        val_text = val_text.tolist()
        test_text = test_text.tolist()
        train_labels = train_labels.tolist()
        val_labels = val_labels.tolist()
        test_labels = test_labels.tolist()
        return train_text, val_text, test_text, train_labels, val_labels, test_labels

    def tokenize_data(self, texts):
        """
        Tokenizes the text data using the BERT tokenizer.

        Args:
            texts (list): The text data to be tokenized.

        Returns:
            dict: A dictionary containing input IDs and attention masks.
        """
        tokens = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_length, padding="max_length", truncation=True
        )
        return tokens

    def create_tensors(self, tokens, labels):
        """
        Converts tokenized data and labels to PyTorch tensors.

        Args:
            tokens (dict): A dictionary containing tokenized data.
            labels (list): The labels corresponding to the data.

        Returns:
            tuple: Containing tensors of input IDs, attention masks, and labels.
        """
        seq = torch.tensor(tokens["input_ids"])
        mask = torch.tensor(tokens["attention_mask"])
        y = torch.tensor(labels)
        return seq, mask, y

    def create_dataloaders(
        self,
        train_seq,
        train_mask,
        train_y,
        val_seq,
        val_mask,
        val_y,
        test_seq,
        test_mask,
        test_y,
        batch_size,
    ):
        """
        Creates DataLoader objects for training, validation, and test sets.

        Args:
            train_seq (torch.Tensor): Tensor of training input IDs.
            train_mask (torch.Tensor): Tensor of training attention masks.
            train_y (torch.Tensor): Tensor of training labels.
            val_seq (torch.Tensor): Tensor of validation input IDs.
            val_mask (torch.Tensor): Tensor of validation attention masks.
            val_y (torch.Tensor): Tensor of validation labels.
            test_seq (torch.Tensor): Tensor of test input IDs.
            test_mask (torch.Tensor): Tensor of test attention masks.
            test_y (torch.Tensor): Tensor of test labels.
            batch_size (int): The batch size for DataLoader.

        Returns:
            tuple: Containing DataLoader for training data,
            DataLoader for validation data, and DataLoader for test data.
        """
        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_seq, val_mask, val_y)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        test_data = TensorDataset(test_seq, test_mask, test_y)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def process(self, batch_size, test_size, val_size, random_state, processed_data_dir):
        """
        Executes the entire preprocessing pipeline from loading data to creating DataLoader objects.

        Args:
            batch_size (int): The batch size for DataLoader.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the test dataset to include in the validation split.
            random_state (int): Random seed for shuffling the data.
            processed_data_dir (str): Directory to load the preprocessed data from.

        Returns:
            tuple: Containing DataLoader for training data, DataLoader for validation data,
            and DataLoader for test data.
        """
        data = self.load_preprocessed_data(processed_data_dir, file_name="preprocessed_data.csv")
        train_text, val_text, test_text, train_labels, val_labels, test_labels = self.split_data(
            data, test_size, val_size, random_state
        )
        tokens_train = self.tokenize_data(train_text)
        tokens_val = self.tokenize_data(val_text)
        tokens_test = self.tokenize_data(test_text)

        train_seq, train_mask, train_y = self.create_tensors(tokens_train, train_labels)
        val_seq, val_mask, val_y = self.create_tensors(tokens_val, val_labels)
        test_seq, test_mask, test_y = self.create_tensors(tokens_test, test_labels)

        return self.create_dataloaders(
            train_seq,
            train_mask,
            train_y,
            val_seq,
            val_mask,
            val_y,
            test_seq,
            test_mask,
            test_y,
            batch_size,
        )

    def create_prediction_dataloader(self, predict_data_dir, batch_size):
        """
        Creates DataLoader object for prediction data.

        Args:
            titles (list): List of titles to be tokenized and converted to DataLoader.
            batch_size (int): The batch size for DataLoader.

        Returns:
            DataLoader: DataLoader object for prediction data.
        """
        # Create DataLoader for prediction data
        predict_data = self.load_preprocessed_data(predict_data_dir, file_name="predict_data.csv")
        titles = predict_data["title"].tolist()
        tokens = self.tokenize_data(titles)
        seq = torch.tensor(tokens["input_ids"])
        mask = torch.tensor(tokens["attention_mask"])
        predict_data = TensorDataset(seq, mask)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=batch_size)
        return predict_dataloader


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function to run the data preprocessing.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    preprocessor = DataPreprocessor(RAW_DATA_DIR, cfg.preprocess.max_length)
    data = preprocessor.load_data()
    preprocessed_data = preprocessor.preprocess_data(data)
    preprocessor.save_preprocessed_data(preprocessed_data, PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
