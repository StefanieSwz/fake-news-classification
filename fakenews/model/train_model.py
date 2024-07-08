from datetime import datetime
import os
import tempfile
import shutil
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
    ModelPruning,
)
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml
import wandb
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass
from fakenews.config import (
    MODELS_DIR,
    access_secret_version,
    setup_data_directories,
    compare_and_upload_best_model,
)


def preprocess_data(cfg: DictConfig, processed_data_dir):
    """
    Preprocess data and return DataLoader objects for training, validation, and test sets.

    This function initializes a DataPreprocessor object and uses it to process the raw data
    into DataLoader objects, which are used for training, validation, and testing a machine learning model.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        processed_data_dir (str): Directory path where the preprocessed data is stored or will be stored.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_dataloader: DataLoader for the training data.
            - val_dataloader: DataLoader for the validation data.
            - test_dataloader: DataLoader for the test data.
    """
    preprocessor = DataPreprocessor(processed_data_dir, cfg.preprocess.max_length)
    train_dataloader, val_dataloader, test_dataloader = preprocessor.process(
        batch_size=cfg.train.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=processed_data_dir,
    )
    return train_dataloader, val_dataloader, test_dataloader


def create_model_directory(cfg: DictConfig, models_dir):
    """
    Create a directory to save the model.

    This function creates a directory to save the model based on the configuration.
    If `save_model` is True in the configuration, a directory is created with the current date and time.
    If `save_model` is False, a temporary directory is created.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        models_dir (str): Base directory where model directories should be created.

    Returns:
        str: The path to the created model directory.
    """
    if cfg.train.save_model:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(models_dir, date_str)
        os.makedirs(model_dir, exist_ok=True)
    else:
        model_dir = tempfile.mkdtemp()
    return model_dir


def train_model(
    cfg: DictConfig,
    model: BERTClass,
    train_dataloader,
    val_dataloader,
    model_dir: str,
    wandb_project,
    wandb_entity,
):
    """
    Train the model.

    This function trains a BERT-based model using PyTorch Lightning's Trainer.
    It sets up callbacks for model checkpointing, early stopping, and a progress bar.
    It also configures the WandbLogger for logging to Weights & Biases.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        model (BERTClass): The BERT-based model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        model_dir (str): Directory where model checkpoints will be saved.
        wandb_project (str): Name of the Weights & Biases project.
        wandb_entity (str): Name of the Weights & Biases entity (user or team).

    Returns:
        None
    """
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=cfg.train.filename,
        save_top_k=1,
        verbose=cfg.train.verbose,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )
    callbacks.append(checkpoint_callback)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.train.patience, verbose=cfg.train.verbose, mode="min"
    )
    callbacks.append(early_stopping_callback)

    progress_bar = TQDMProgressBar(refresh_rate=cfg.train.refresh_rate)
    callbacks.append(progress_bar)

    if cfg.train.pruning:
        pruning_callback = ModelPruning("l1_unstructured", amount=0.3)
        callbacks.append(pruning_callback)

    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    wandb_logger = WandbLogger(
        log_model=cfg.train.log_model,
        project=wandb_project,
        entity=wandb_entity,
    )

    trainer = Trainer(
        profiler=cfg.train.profiler,
        precision=cfg.train.precision,
        max_epochs=cfg.train.epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_checkpointing=True,
        enable_model_summary=True,
        logger=wandb_logger,
        default_root_dir=model_dir,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f"Callback metrics: {trainer.callback_metrics}")
    val_loss = trainer.callback_metrics.get("val_loss")
    print(f"Validation loss: {val_loss}")
    if val_loss is None:
        raise ValueError("Validation loss not found in callback metrics. Ensure it is being logged during validation.")

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    return best_model_path, val_loss.item()


def eval_model(cfg: DictConfig, model_dir: str, test_dataloader, wandb_project, wandb_entity):
    """
    Evaluate the model.

    This function loads a BERT-based model from a checkpoint, evaluates it on a test dataset,
    and logs the results to Weights & Biases.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        model_dir (str): Directory where model checkpoints are saved.
        test_dataloader (DataLoader): DataLoader for the test data.
        wandb_project (str): Name of the Weights & Biases project.
        wandb_entity (str): Name of the Weights & Biases entity (user or team).

    Returns:
        float: The test loss of the model.
    """
    model_checkpoint_path = os.path.join(model_dir, cfg.train.filename + ".ckpt")
    model = BERTClass.load_from_checkpoint(model_checkpoint_path, cfg=cfg)
    print(f"Loaded model from checkpoint: {model_checkpoint_path}")

    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
    )

    trainer = Trainer(
        accelerator=("gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        devices=cfg.train.devices,
        logger=wandb_logger,
    )

    result = trainer.test(model, dataloaders=test_dataloader)
    for key, value in result[0].items():
        wandb.log({key: value})
    return result[0]["test_loss"]


def run_sweep(cfg: DictConfig, processed_data_dir, models_dir, wandb_project, wandb_entity):
    """
    Run a Weights & Biases sweep.

    This function loads a sweep configuration from a YAML file, initializes a Weights & Biases (wandb) sweep,
    and defines a training function to be called by the wandb agent for training the model.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        processed_data_dir (str): Directory path where the preprocessed data is stored or will be stored.
        models_dir (str): Base directory where model directories should be created.
        wandb_project (str): Name of the Weights & Biases project.
        wandb_entity (str): Name of the Weights & Biases entity (user or team).

    Returns:
        None
    """
    # Load sweep configuration from YAML file
    with open(os.path.join(os.path.dirname(__file__), "../../config/sweep.yaml"), "r") as file:
        sweep_config = yaml.safe_load(file)
        print(sweep_config)

    # Initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config, project=wandb_project, entity=wandb_entity)

    def train():
        """
        Function to be called by wandb agent for training.

        Args:
            None

        Returns:
            None
        """
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

        # Create model directory
        model_dir = create_model_directory(cfg, models_dir)

        # Preprocess data
        train_dataloader, val_dataloader, test_dataloader = preprocess_data(cfg, processed_data_dir)

        # Initialize the model
        model = BERTClass(cfg)

        # Train the model
        model_checkpoint_path, val_loss = train_model(
            cfg, model, train_dataloader, val_dataloader, model_dir, wandb_project, wandb_entity
        )

        if cfg.cloud.save_best_model_gcs:
            compare_and_upload_best_model(cfg, model_checkpoint_path, val_loss)
        else:
            print("Cloud saving is disabled in the configuration. Not uploading model to GCS.")

        # Evaluate the model
        eval_model(cfg, model_dir, test_dataloader, wandb_project, wandb_entity)

        # Remove temporary model directory if not saving
        if not cfg.train.save_model:
            shutil.rmtree(model_dir)

    # Run wandb agent
    wandb.agent(sweep_id, function=train, count=cfg.train.num_runs)


def train_fixed(cfg: DictConfig, processed_data_dir, models_dir, wandb_api_key, wandb_project, wandb_entity):
    """
    Train the model with fixed configuration.

    This function trains a BERT-based model using a fixed configuration. It logs into Weights & Biases,
    initializes a new run, preprocesses the data, trains the model, evaluates the model, and logs the results to Weights & Biases.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        processed_data_dir (str): Directory path where the preprocessed data is stored or will be stored.
        models_dir (str): Base directory where model directories should be created.
        wandb_api_key (str): API key for authenticating with Weights & Biases.
        wandb_project (str): Name of the Weights & Biases project.
        wandb_entity (str): Name of the Weights & Biases entity (user or team).

    Returns:
        None
    """
    wandb.login(key=wandb_api_key)
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
    )

    # Print the configuration
    print(f"Training Config: {OmegaConf.to_container(cfg, resolve=True)}")

    # Create model directory
    model_dir = create_model_directory(cfg, models_dir)

    # Preprocess data
    train_dataloader, val_dataloader, test_dataloader = preprocess_data(cfg, processed_data_dir)

    # Initialize the model
    model = BERTClass(cfg)

    # Train the model
    model_checkpoint_path, val_loss = train_model(
        cfg, model, train_dataloader, val_dataloader, model_dir, wandb_project, wandb_entity
    )
    if cfg.cloud.save_best_model_gcs:
        compare_and_upload_best_model(cfg, model_checkpoint_path, val_loss)
    else:
        print("Cloud saving is disabled in the configuration. Not uploading model to GCS.")

    # Evaluate the model
    eval_model(cfg, model_dir, test_dataloader, wandb_project, wandb_entity)

    # Remove temporary model directory if not saving
    if not cfg.train.save_model:
        shutil.rmtree(model_dir)


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main function to handle command-line arguments and run appropriate training function."""
    if cfg.train.local_wandb:
        from fakenews.config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
    else:
        # Fetch secrets using Secret Manager if they are not set
        WANDB_API_KEY = access_secret_version("WANDB_API_KEY")
        WANDB_PROJECT = access_secret_version("WANDB_PROJECT")
        WANDB_ENTITY = access_secret_version("WANDB_ENTITY")

    # Validate that all required environment variables are present
    if not WANDB_API_KEY:
        raise ValueError("WANDB_API_KEY environment variable is not set.")
    if not WANDB_PROJECT:
        raise ValueError("WANDB_PROJECT environment variable is not set.")
    if not WANDB_ENTITY:
        raise ValueError("WANDB_ENTITY environment variable is not set.")

    if cfg.train.local_data:
        from fakenews.config import PROCESSED_DATA_DIR
    else:
        _, PROCESSED_DATA_DIR, _, _ = setup_data_directories(cfg=cfg)

    if cfg.train.sweep:
        run_sweep(cfg, PROCESSED_DATA_DIR, MODELS_DIR, WANDB_PROJECT, WANDB_ENTITY)
    else:
        train_fixed(cfg, PROCESSED_DATA_DIR, MODELS_DIR, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY)


if __name__ == "__main__":
    main()
