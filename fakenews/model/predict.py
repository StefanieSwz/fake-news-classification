import os
import hydra
from omegaconf import DictConfig
import torch
import wandb

from fakenews.config import (
    MODEL_REGISTRY,
    PREDICT_DATA_DIR,
    WANDB_API_KEY,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def predict(cfg: DictConfig):
    """
    Predict the labels for new instances using the trained model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """

    # Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    # Use the model artifact from wandb
    artifact = run.use_artifact(f"{WANDB_ENTITY}/model-registry/{MODEL_REGISTRY}:best", type="model")
    artifact_dir = artifact.download()

    # Initialize the DataPreprocessor
    preprocessor = DataPreprocessor(data_dir=None, max_length=cfg.preprocess.max_length)

    # Create DataLoader for prediction data
    predict_dataloader = preprocessor.create_prediction_dataloader(PREDICT_DATA_DIR, batch_size=cfg.train.batch_size)

    # Load the trained model
    model = BERTClass.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), cfg=cfg)

    # Determine the device (CPU, GPU, MPS)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Move model to the device
    model.to(device)
    model.eval()

    # Predict
    predictions = []
    with torch.no_grad():
        for batch in predict_dataloader:
            sent_id, mask = [t.to(device) for t in batch]
            outputs = model(sent_id=sent_id, mask=mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.tolist())

    # Print predictions
    print(predictions)


if __name__ == "__main__":
    predict()
