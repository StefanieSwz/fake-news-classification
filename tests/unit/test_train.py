"""
This script tests the model training.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from hydra import compose, initialize
import torch

from fakenews.model.model import BERTClass
from fakenews.model.train_model import (
    preprocess_data,
    create_model_directory,
    train_model,
    eval_model,
    run_sweep,
)


@pytest.fixture
def cfg():
    """Fixture to initialize Hydra and load the configuration."""
    with initialize(config_path="../../config", version_base="1.2"):
        cfg = compose(config_name="config")
    return cfg


@pytest.fixture
def sweep_config():
    """Fixture to provide a sweep configuration for testing."""
    return {
        "train.lr": 5e-5,
        "train.batch_size": 16,
        "model.dropout_rate": 0.1,
    }


@patch("fakenews.data.preprocessing.DataPreprocessor.process")
def test_preprocess_data(mock_process, cfg):
    """
    Test the preprocess_data function.

    Args:
        mock_process: Mock object for DataPreprocessor.process.
        cfg: Configuration object.
    """
    mock_process.return_value = (None, None, None)
    train_dataloader, val_dataloader, test_dataloader = preprocess_data(cfg, "processed_data_dir")
    assert train_dataloader is None
    assert val_dataloader is None
    assert test_dataloader is None


def test_create_model_directory(cfg):
    """
    Test the create_model_directory function.

    Args:
        cfg: Configuration object.
    """
    model_dir = create_model_directory(cfg, "models_dir")
    assert os.path.exists(model_dir)
    if cfg.train.save_model:
        assert model_dir.startswith("models_dir")
    else:
        assert model_dir.startswith(tempfile.gettempdir())


@patch("pytorch_lightning.Trainer.fit")
def test_train_model(mock_trainer_fit, cfg):
    """
    Test the train_model function.

    Args:
        mock_trainer_fit: Mock object for Trainer.fit.
        cfg: Configuration object.
    """
    model = BERTClass(cfg)
    train_dataloader, val_dataloader = MagicMock(), MagicMock()
    with patch("pytorch_lightning.Trainer.callback_metrics", {"val_loss": torch.tensor(0.1)}):
        best_model_path, val_loss = train_model(
            cfg,
            model,
            train_dataloader,
            val_dataloader,
            "model_dir",
            "wandb_project",
            "wandb_entity",
        )
    mock_trainer_fit.assert_called_once()
    assert best_model_path is not None
    assert val_loss == torch.tensor(0.1)


@patch("wandb.init")
@patch("wandb.log")
@patch("pytorch_lightning.Trainer.test")
@patch("pytorch_lightning.loggers.WandbLogger")
@patch("fakenews.model.model.BERTClass.load_from_checkpoint")
def test_eval_model(
    mock_load_from_checkpoint,
    mock_wandb_logger,
    mock_trainer_test,
    mock_wandb_log,
    mock_wandb_init,
    cfg,
):
    """
    Test the eval_model function.

    Args:
        mock_load_from_checkpoint: Mock object for BERTClass.load_from_checkpoint.
        mock_wandb_logger: Mock object for WandbLogger.
        mock_trainer_test: Mock object for Trainer.test.
        mock_wandb_log: Mock object for wandb.log.
        mock_wandb_init: Mock object for wandb.init.
        cfg: Configuration object.
    """
    mock_load_from_checkpoint.return_value = BERTClass(cfg)
    mock_trainer_test.return_value = [{"test_loss": 0.1}]
    mock_wandb_init.return_value = MagicMock()
    test_loss = eval_model(cfg, "model_dir", MagicMock(), "wandb_project", "wandb_entity")
    assert test_loss == 0.1
    mock_trainer_test.assert_called_once()


@patch("wandb.sweep")
@patch("wandb.agent")
@patch("wandb.login")  # Add this line
def test_run_sweep(mock_login, mock_agent, mock_sweep, cfg):
    """
    Test the run_sweep function.

    Args:
        mock_login: Mock object for wandb.login.
        mock_agent: Mock object for wandb.agent.
        mock_sweep: Mock object for wandb.sweep.
        cfg: Configuration object.
    """
    mock_sweep.return_value = "sweep_id"
    mock_login.return_value = True  # Mock the login method
    run_sweep(cfg, "processed_data_dir", "models_dir", "wandb_api_key", "wandb_project", "wandb_entity")
    mock_agent.assert_called_once()
