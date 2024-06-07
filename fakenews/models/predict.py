import os
import torch
from pytorch_lightning import Trainer
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.config import PROCESSED_DATA_DIR, MODELS_DIR
from fakenews.models.model import BERTClass
import logging
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig):
    """
    Evaluate a trained BERT model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    log = logging.getLogger(__name__)

    model_checkpoint_path = os.path.join(to_absolute_path(MODELS_DIR), cfg.eval.model_checkpoint)
    log.info(f"Model checkpoint: {model_checkpoint_path}")

    # Initialize the model with the cfg and load the checkpoint state
    model = BERTClass(cfg)
    model.load_state_dict(torch.load(model_checkpoint_path)["state_dict"])

    preprocessor = DataPreprocessor(to_absolute_path(PROCESSED_DATA_DIR), max_length=cfg.preprocess.max_length)
    _, _, test_dataloader = preprocessor.process(
        batch_size=cfg.eval.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=to_absolute_path(PROCESSED_DATA_DIR),
    )

    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    wandb_logger = WandbLogger(
        log_model="all",
        project="fake-news-classification",
        entity="mlops-fakenews",  # Ensure this is correct and you have access
        config={
            "lr": cfg.train.lr,
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "patience": cfg.train.patience,
        },
    )

    trainer = Trainer(accelerator=accelerator, devices=devices, logger=wandb_logger)
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    evaluate()
