import os
import sys

import click
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from fakenews.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.models.main import train_and_eval

# Suppress the Huggingface tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def preprocess(cfg: DictConfig):
    """Hydra-decorated function for data preprocessing."""
    preprocessor = DataPreprocessor(RAW_DATA_DIR, cfg.preprocess.max_length)
    data = preprocessor.load_data()
    preprocessed_data = preprocessor.preprocess_data(data)
    preprocessor.save_preprocessed_data(preprocessed_data, PROCESSED_DATA_DIR)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train_and_evaluate_fun(cfg: DictConfig):
    """Hydra-decorated function to train and evaluate the BERT model."""
    train_and_eval(cfg)


@click.group()
def cli():
    """Command line interface for preprocessing, training, and evaluating the BERT model."""
    pass


@cli.command(name="preprocess")
@click.pass_context
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def preprocess_cmd(ctx, overrides):
    """
    Command to preprocess the data.

    This command clears the Click arguments and invokes the `preprocess` function.
    """
    sys.argv = [sys.argv[0]] + list(overrides)  # Clear the Click arguments and pass the remaining to Hydra
    preprocess()


@cli.command(name="train_and_eval")
@click.pass_context
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train_and_evaluate_cmd(ctx, overrides):
    """
    Command to train and evaluate the model.

    This command clears the Click arguments and invokes the `train_and_evaluate_fun` function.
    """
    sys.argv = [sys.argv[0]] + list(overrides)  # Clear the Click arguments and pass the remaining to Hydra
    train_and_evaluate_fun()


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
