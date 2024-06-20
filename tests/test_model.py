"""This script tests the models."""
import sys
import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
#import pytest
#import pytest
#import torch
#from omegaconf import OmegaConf, DictConfig
#from transformers import AutoTokenizer
#from torch.utils.data import DataLoader, TensorDataset
#from fakenews.models.model import BERTClass
#from hydra import initialize, compose
#import pytorch_lightning as pl
#from transformers import AutoModel
#
## Load the configuration from the YAML file
##cfg = OmegaConf.load('config/model/model.yaml')
#
## Initialize Hydra and load the configuration
#@pytest.fixture
#def cfg():
#    with initialize(config_path="../config", version_base=None):
#        cfg = compose(config_name="config")
#    return cfg
#
#@pytest.fixture
#def model(cfg):
#    return BERTClass(cfg)
#
#
#def test_model_initialization(cfg):
#    model = BERTClass(cfg)
#    assert model.dropout.p == cfg.model.dropout_rate
#
#