import click
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import initialize, compose
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from fakenews.models.evaluate import evaluate
from fakenews.models.train import train


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def preprocess(cfg: DictConfig):
    """
    Hydra-decorated function for data preprocessing.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    preprocessor = DataPreprocessor(RAW_DATA_DIR, cfg.preprocess.max_length)
    data = preprocessor.load_data()
    preprocessed_data = preprocessor.preprocess_data(data)
    preprocessor.save_preprocessed_data(preprocessed_data, PROCESSED_DATA_DIR)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def evaluate_fun(cfg: DictConfig):
    """
    Hydra-decorated function to evaluate a trained BERT model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    evaluate(cfg)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train_fun(cfg: DictConfig):
    """
    Hydra-decorated function to train the BERT model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    train(cfg)


@click.group()
def cli():
    """Command line interface for preprocessing, training, and evaluating the BERT model."""
    pass


@cli.command(name="preprocess")
def preprocess_cmd():
    """
    Command to preprocess the data.

    This command clears the Click arguments and invokes the `preprocess` function.
    """
    sys.argv = [sys.argv[0]]  # Clear the Click arguments
    preprocess()


@cli.command(name="evaluate")
def evaluate_cmd():
    """
    Command to evaluate the model.

    This command clears the Click arguments and invokes the `evaluate_fun` function.
    """
    sys.argv = [sys.argv[0]]  # Clear the Click arguments
    evaluate_fun()


@cli.command(name="train")
def train_cmd():
    """
    Command to train the model.

    This command clears the Click arguments and invokes the `train_fun` function.
    """
    sys.argv = [sys.argv[0]]  # Clear the Click arguments
    train_fun()


@cli.command()
def entries():
    """
    Command to print the composed configuration.

    This command initializes Hydra and composes the configuration, then prints it.
    """
    with initialize(config_path="config", version_base="1.2"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    cli()
