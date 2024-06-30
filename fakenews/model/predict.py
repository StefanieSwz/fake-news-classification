import os
import hydra
from omegaconf import DictConfig
import torch
import wandb
from google.cloud import secretmanager, storage
import tempfile

from fakenews.config import (
    MODEL_REGISTRY,
)
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass


def access_secret_version(secret_id):
    """Access the latest version of a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = "mlops-fakenews"  # Replace with your project ID
    secret_version_id = "latest"
    secret_version_name = f"projects/{project_id}/secrets/{secret_id}/versions/{secret_version_id}"
    response = client.access_secret_version(name=secret_version_name)
    return response.payload.data.decode("UTF-8")


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


def get_blob_from_gcs(bucket_name, blob_name):
    """Fetch a blob from Google Cloud Storage and return it as bytes."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def setup_data_directories():
    """Fetch data from GCS and set up temporary directories for data."""
    # Create temporary directory
    predict_data_dir = tempfile.mkdtemp()
    # Access predict data from Google Cloud Storage
    predict_data = get_blob_from_gcs("mlops-lmu-data-bucket", "data/predict/predict_data.csv")

    # Save predict data to temporary files
    predict_data_path = os.path.join(predict_data_dir, "predict_data.csv")

    with open(predict_data_path, "wb") as f:
        f.write(predict_data)

    return predict_data_dir


PREDICT_DATA_DIR = setup_data_directories()


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
