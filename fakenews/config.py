import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from google.cloud import secretmanager, storage

import tempfile


# Load environment variables from .env file if it exists
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
MODEL_REGISTRY = "models"

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATASET = "saurabhshahane/fake-news-classification"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PREDICT_DATA_DIR = DATA_DIR / "predict"

MODELS_DIR = PROJ_ROOT / "models"
BEST_MODEL = MODELS_DIR / "best_model"
DEPLOY_MODEL = MODELS_DIR / "deploy"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def access_secret_version(secret_id):
    """Access the latest version of a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = "mlops-fakenews"  # Replace with your project ID
    secret_version_id = "latest"
    secret_version_name = f"projects/{project_id}/secrets/{secret_id}/versions/{secret_version_id}"
    response = client.access_secret_version(name=secret_version_name)
    return response.payload.data.decode("UTF-8")


def get_blob_from_gcs(bucket_name, blob_name):
    """Fetch a blob from Google Cloud Storage and return it as bytes."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def setup_data_directories():
    """Fetch data from GCS and set up temporary directories for data."""
    # Create temporary directories
    raw_data_dir = tempfile.mkdtemp()
    processed_data_dir = tempfile.mkdtemp()
    predict_data_dir = tempfile.mkdtemp()

    # Access raw and processed data from Google Cloud Storage
    raw_data = get_blob_from_gcs("mlops-lmu-data-bucket", "data/raw/fake-news-classification.zip")
    processed_data = get_blob_from_gcs("mlops-lmu-data-bucket", "data/processed/preprocessed_data.csv")
    predict_data = get_blob_from_gcs("mlops-lmu-data-bucket", "data/predict/predict_data.csv")

    # Save raw and processed data to temporary files
    raw_data_path = os.path.join(raw_data_dir, "fake-news-classification.zip")
    processed_data_path = os.path.join(processed_data_dir, "preprocessed_data.csv")
    predict_data_path = os.path.join(predict_data_dir, "predict_data.csv")

    with open(raw_data_path, "wb") as f:
        f.write(raw_data)

    with open(processed_data_path, "wb") as f:
        f.write(processed_data)

    with open(predict_data_path, "wb") as f:
        f.write(predict_data)

    return predict_data_dir, processed_data_dir, raw_data_dir


# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
