"""This script tests the models."""

from hydra import compose, initialize
import pytest

from fakenews.model.model import BERTClass

# Load the configuration from the YAML file
# cfg = OmegaConf.load('config/model/model.yaml')


# Initialize Hydra and load the configuration
@pytest.fixture
def cfg():
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="config")
    return cfg


@pytest.fixture
def model(cfg):
    return BERTClass(cfg)


def test_model_initialization(cfg):
    model = BERTClass(cfg)
    assert model.dropout.p == cfg.model.dropout_rate
