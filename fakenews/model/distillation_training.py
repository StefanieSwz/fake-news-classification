import os
import shutil
import tempfile
import hydra
from omegaconf import DictConfig
import wandb
from fakenews.model.model import BERTClass
from fakenews.model.distillation_model import DistillationTrainer, StudentDistilBERTClass
from fakenews.config import (
    DISTILLED_MODEL,
    MODEL_REGISTRY,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
    PROCESSED_DATA_DIR,
    upload_to_gcs,
)
from fakenews.model.train_model import preprocess_data, train_model, eval_model


def get_hyperparameters_from_wandb(artifact_path):
    """
    Retrieve hyperparameters from a WandB artifact.

    Args:
        artifact_path (str): The path to the artifact in WandB.

    Returns:
        dict: A dictionary containing the hyperparameters.
    """
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    run = artifact.logged_by()
    config = run.config
    return config


def train_distillation(
    cfg: DictConfig,
    teacher_model: BERTClass,
    student_model: StudentDistilBERTClass,
    train_dataloader,
    val_dataloader,
    model_dir: str,
    wandb_project,
    wandb_entity,
):
    """
    Train the student model using knowledge distillation.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        teacher_model (BERTClass): The pre-trained teacher model.
        student_model (StudentBERTClass): The student model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        model_dir (str): Directory where model checkpoints will be saved.
        wandb_project (str): Name of the Weights & Biases project.
        wandb_entity (str): Name of the Weights & Biases entity (user or team).

    Returns:
        tuple: Best model path and validation loss.
    """
    distillation_trainer = DistillationTrainer(teacher_model, student_model, cfg)

    # Reuse the existing train_model function
    best_model_path, val_loss = train_model(
        cfg, distillation_trainer, train_dataloader, val_dataloader, model_dir, wandb_project, wandb_entity
    )
    return best_model_path, val_loss


def train_student(cfg: DictConfig, processed_data_dir, wandb_project, wandb_entity):
    """
    Train the student model using fixed configuration.

    This function sets up the data, loads the teacher model, initializes the student model,
    trains the student model using knowledge distillation, and evaluates the student model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        processed_data_dir (str): Directory where the processed data is stored.
        models_dir (str): Directory where model checkpoints will be saved.
        wandb_project (str): Name of the Weights & Biases project.
        wandb_entity (str): Name of the Weights & Biases entity (user or team).

    Returns:
        None
    """
    # Use Wandb with local credentials
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=wandb_project, entity=wandb_entity)

    model_dir = DISTILLED_MODEL
    os.makedirs(model_dir, exist_ok=True)
    train_dataloader, val_dataloader, test_dataloader = preprocess_data(cfg, processed_data_dir)

    # Load the teacher model
    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact = run.use_artifact(f"{wandb_entity}/model-registry/{MODEL_REGISTRY}:best", type="model")
        artifact.download(root=artifact_dir)
        teacher_model_path = os.path.join(artifact_dir, "model.ckpt")
        teacher_model = BERTClass.load_from_checkpoint(teacher_model_path, cfg=cfg)

        # Extract hyperparameters from the teacher model wandb run
        teacher_hyperparams = get_hyperparameters_from_wandb(f"{wandb_entity}/model-registry/{MODEL_REGISTRY}:best")

        # Print the hyperparameters to understand their structure
        print("Teacher Hyperparameters:", teacher_hyperparams)

        # Use .get() to safely access dictionary keys
        cfg.train.lr = teacher_hyperparams.get("train.lr", cfg.train.lr)
        cfg.model.dropout_rate = teacher_hyperparams.get("model.dropout_rate", cfg.model.dropout_rate)
        cfg.train.batch_size = teacher_hyperparams.get("train.batch_size", cfg.train.batch_size)
        cfg.train.pruning_rate = teacher_hyperparams.get("train.pruning_rate", 0.0)  # Default to 0.0 if not present

    # Initialize the student model with the same hyperparameters
    student_model = StudentDistilBERTClass(cfg)

    # Train the student model
    best_model_path, _ = train_distillation(
        cfg, teacher_model, student_model, train_dataloader, val_dataloader, model_dir, wandb_project, wandb_entity
    )

    # Save the model checkpoint with a fixed name
    fixed_checkpoint_path = os.path.join(model_dir, "model.ckpt")
    shutil.copy(best_model_path, fixed_checkpoint_path)

    # Evaluate the student model
    eval_model(cfg, model_dir, test_dataloader, wandb_project, wandb_entity, model_class=StudentDistilBERTClass)

    # Upload the best model to GCS
    with open(fixed_checkpoint_path, "rb") as f:
        upload_to_gcs(f, cfg.cloud.bucket_name_model, os.path.join(cfg.cloud.model_dir, cfg.cloud.distilled_file))

    if not cfg.train.save_model:
        shutil.rmtree(model_dir)


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function to handle command-line arguments and run the training process.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    train_student(cfg, PROCESSED_DATA_DIR, WANDB_PROJECT, WANDB_ENTITY)


if __name__ == "__main__":
    main()
