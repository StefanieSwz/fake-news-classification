import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.config import PROCESSED_DATA_DIR, MODELS_DIR
from fakenews.config import WANDB_PROJECT, WANDB_ENTITY
from fakenews.models.model import BERTClass
from datetime import datetime
import hydra


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def train_and_eval(cfg: DictConfig):
    """
    Function to train and evaluate the BERT model using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # Create a dated directory within MODELS_DIR
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODELS_DIR, date_str)
    os.makedirs(model_dir, exist_ok=True)

    preprocessor = DataPreprocessor(PROCESSED_DATA_DIR, cfg.preprocess.max_length)
    train_dataloader, val_dataloader, test_dataloader = preprocessor.process(
        batch_size=cfg.main.batch_size,
        test_size=cfg.main.test_size,
        val_size=cfg.main.val_size,
        random_state=cfg.main.random_state,
        processed_data_dir=PROCESSED_DATA_DIR,
    )

    model = BERTClass(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=cfg.main.filename,
        save_top_k=1,
        verbose=cfg.main.verbose,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,  # Ensure the entire model is saved, including hyperparameters
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.main.patience, verbose=cfg.main.verbose, mode="min"
    )

    progress_bar = TQDMProgressBar(refresh_rate=cfg.main.refresh_rate)

    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    wandb_logger = WandbLogger(
        log_model="all",
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )

    trainer = Trainer(
        profiler=cfg.main.profiler,
        precision=cfg.main.precision,
        max_epochs=cfg.main.epochs,
        callbacks=[early_stopping_callback, checkpoint_callback, progress_bar],
        accelerator=accelerator,
        devices=cfg.main.devices,
        log_every_n_steps=cfg.main.log_every_n_steps,
        enable_checkpointing=True,
        enable_model_summary=True,
        logger=wandb_logger,
    )
    print(f"Config: {cfg}")

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Load the best model for evaluation
    model_checkpoint_path = os.path.join(model_dir, cfg.main.filename + ".ckpt")
    model = BERTClass.load_from_checkpoint(model_checkpoint_path)

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    train_and_eval()
