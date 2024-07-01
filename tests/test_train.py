"""This script tests the model training."""

import pytest
from unittest.mock import patch
from hydra import compose, initialize

from fakenews.model.train_model import (
    update_config_with_sweep,
    run_sweep,
)


# Initialize Hydra and load the configuration
@pytest.fixture
def cfg():
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="config")
    return cfg


@pytest.fixture
def sweep_config():
    return {
        "train.lr": 5e-5,
        "train.batch_size": 16,
        "model.dropout_rate": 0.1,
    }


def test_update_config_with_sweep(cfg, sweep_config):
    updated_cfg = update_config_with_sweep(cfg, sweep_config)
    assert updated_cfg.train.lr == sweep_config["train.lr"]
    assert updated_cfg.train.batch_size == sweep_config["train.batch_size"]
    assert updated_cfg.model.dropout_rate == sweep_config["model.dropout_rate"]


@patch("wandb.init")
@patch("wandb.sweep")
@patch("wandb.agent")
def test_run_sweep(mock_agent, mock_sweep, mock_init, cfg):
    mock_sweep.return_value = "test-sweep-id"
    mock_agent.side_effect = lambda *args, **kwargs: None
    run_sweep(cfg)
    mock_sweep.assert_called_once()
    mock_agent.assert_called_once()
