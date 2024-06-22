from datetime import datetime
import os

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml

from fakenews.config import MODELS_DIR, PROCESSED_DATA_DIR, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass
import wandb


def update_config_with_sweep(cfg: DictConfig, sweep_config):
    """Update Hydra configuration with sweep parameters.

    Args:
        cfg (DictConfig): Original Hydra configuration.
        sweep_config (dict): Sweep configuration from wandb.

    Returns:
        DictConfig: Updated Hydra configuration.
    """
    cfg.train.lr = sweep_config["train.lr"]
    cfg.train.batch_size = sweep_config["train.batch_size"]
    cfg.model.dropout_rate = sweep_config["model.dropout_rate"]
    return cfg


def run_sweep(cfg: DictConfig):
    """Run a wandb sweep.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    # Load sweep configuration from YAML file
    with open(os.path.join(os.path.dirname(__file__), "../../config/sweep.yaml"), "r") as file:
        sweep_config = yaml.safe_load(file)
        print(sweep_config)

    # Initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)

    def train():
        """Function to be called by wandb agent for training."""
        wandb.init()
        sweep_config = wandb.config

        # Clear any existing Hydra instance
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Hydra config initialization
        with hydra.initialize(config_path="../../config", version_base="1.2"):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"train.lr={sweep_config['train.lr']}",
                    f"train.batch_size={sweep_config['train.batch_size']}",
                    f"model.dropout_rate={sweep_config['model.dropout_rate']}",
                ],
            )

        # Print the updated configuration
        print(f"Training Config: {OmegaConf.to_container(cfg, resolve=True)}")

        # Create a dated directory within MODELS_DIR
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(MODELS_DIR, date_str)
        os.makedirs(model_dir, exist_ok=True)

        # Preprocess data
        preprocessor = DataPreprocessor(PROCESSED_DATA_DIR, cfg.preprocess.max_length)
        train_dataloader, val_dataloader, test_dataloader = preprocessor.process(
            batch_size=cfg.train.batch_size,
            test_size=cfg.train.test_size,
            val_size=cfg.train.val_size,
            random_state=cfg.train.random_state,
            processed_data_dir=PROCESSED_DATA_DIR,
        )

        model = BERTClass(cfg)

        # Train the model
        train_model(cfg, model, train_dataloader, val_dataloader, model_dir)

        # Evaluate the model
        eval_model(cfg, model_dir, test_dataloader)

    wandb.agent(sweep_id, function=train, count=cfg.train.num_runs)


def train_fixed(cfg: DictConfig):
    """Train the model with fixed configuration.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "train.lr": cfg.train.lr,
            "train.batch_size": cfg.train.batch_size,
            "train.epochs": cfg.train.epochs,
            "model.dropout_rate": cfg.model.dropout_rate,
        },
    )
    config = wandb.config

    # Merge wandb.config into the Hydra cfg
    cfg = update_config_with_sweep(cfg, config)

    # Print the updated configuration
    print(f"Training Config: {OmegaConf.to_container(cfg, resolve=True)}")

    # Create a dated directory within MODELS_DIR
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODELS_DIR, date_str)
    os.makedirs(model_dir, exist_ok=True)

    # Preprocess data
    preprocessor = DataPreprocessor(PROCESSED_DATA_DIR, cfg.preprocess.max_length)
    train_dataloader, val_dataloader, test_dataloader = preprocessor.process(
        batch_size=cfg.train.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=PROCESSED_DATA_DIR,
    )

    model = BERTClass(cfg)

    # Train the model
    train_model(cfg, model, train_dataloader, val_dataloader, model_dir)

    # Evaluate the model
    eval_model(cfg, model_dir, test_dataloader)


def train_model(cfg: DictConfig, model: BERTClass, train_dataloader, val_dataloader, model_dir: str):
    """Train the model.

    Args:
        cfg (DictConfig): Hydra configuration.
        model (BERTClass): Model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        model_dir (str): Directory to save model checkpoints.
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=cfg.train.filename,
        save_top_k=1,
        verbose=cfg.train.verbose,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.train.patience, verbose=cfg.train.verbose, mode="min"
    )

    progress_bar = TQDMProgressBar(refresh_rate=cfg.train.refresh_rate)

    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    wandb_logger = WandbLogger(
        log_model="all",
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )

    trainer = Trainer(
        profiler=cfg.train.profiler,
        precision=cfg.train.precision,
        max_epochs=cfg.train.epochs,
        callbacks=[early_stopping_callback, checkpoint_callback, progress_bar],
        accelerator=accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_checkpointing=True,
        enable_model_summary=True,
        logger=wandb_logger,
    )
    print(f"Training Config: {cfg}")

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def eval_model(cfg: DictConfig, model_dir: str, test_dataloader):
    """Evaluate the model.

    Args:
        cfg (DictConfig): Hydra configuration.
        model_dir (str): Directory where the model checkpoints are saved.
        test_dataloader (DataLoader): DataLoader for test data.
    """
    model_checkpoint_path = os.path.join(model_dir, cfg.train.filename + ".ckpt")
    model = BERTClass.load_from_checkpoint(model_checkpoint_path)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        devices=cfg.train.devices,
    )

    trainer.test(model, dataloaders=test_dataloader)


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main function to handle command-line arguments and run appropriate training function.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    if cfg.train.sweep:
        run_sweep(cfg)
    else:
        train_fixed(cfg)


if __name__ == "__main__":
    main()
