"""This script tests the models."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize
import pytest

try:
    from fakenews.model.model import BERTClass
except Exception as e:
    print(e)
    from fakenews.models.model import BERTClass

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
