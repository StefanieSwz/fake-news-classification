"""This script tests the models."""

from unittest.mock import MagicMock
from hydra import compose, initialize
import pytest
import torch

from fakenews.model.model import BERTClass

# Load the configuration from the YAML file
# cfg = OmegaConf.load('config/model/model.yaml')


# Initialize Hydra and load the configuration
@pytest.fixture
def cfg():
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="config")
    return cfg


@pytest.fixture
def model(cfg):
    return BERTClass(cfg)


@pytest.fixture
def sample_batch():
    # Create a sample batch of data
    sent_id = torch.tensor([[101, 2054, 2003, 1996, 2568, 102], [101, 2129, 2079, 2017, 4067, 102]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    labels = torch.tensor([1, 0])
    return sent_id, mask, labels


def test_model_initialization(cfg):
    model = BERTClass(cfg)
    assert isinstance(model, BERTClass)
    assert model.bert.config.name_or_path == cfg.model.name


def test_forward_pass(model, sample_batch, cfg):
    sent_id, mask, _ = sample_batch

    # Perform the forward pass
    output = model(sent_id, mask)

    # Check that the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor"

    # Check the shape of the output
    expected_shape = (sent_id.size(0), cfg.model.output_size)
    assert output.shape == expected_shape, f"Output shape is {output.shape}, expected {expected_shape}"

    # Check the type of the output
    assert output.dtype == torch.float32, "Output dtype is not float32"


def test_training_step(model, sample_batch):
    sent_id, mask, labels = sample_batch
    batch = (sent_id, mask, labels)
    batch_idx = 0
    model.log = MagicMock()
    loss = model.training_step(batch, batch_idx)
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"


def test_validation_step(model, sample_batch):
    batch = sample_batch
    batch_idx = 0
    val_output = model.validation_step(batch, batch_idx)
    assert "val_loss" in val_output
    assert "val_acc" in val_output


def test_test_step(model, sample_batch):
    batch = sample_batch
    batch_idx = 0
    test_output = model.test_step(batch, batch_idx)
    assert "loss" in test_output
    assert "acc" in test_output


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)


def test_get_preds_loss_accuracy(model, sample_batch, cfg):
    preds, loss, acc = model._get_preds_loss_accuracy(sample_batch)
    assert preds.shape == (2, cfg.model.output_size)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(acc, torch.Tensor)
