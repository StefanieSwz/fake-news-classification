import os
from pathlib import Path
import io

from dotenv import load_dotenv
from loguru import logger
import shutil

from omegaconf import DictConfig
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


def setup_data_directories(cfg: DictConfig):
    """Fetch data from GCS and set up temporary directories for data."""
    # Create temporary directories
    raw_data_dir = tempfile.mkdtemp()
    processed_data_dir = tempfile.mkdtemp()
    predict_data_dir = tempfile.mkdtemp()

    # Access raw and processed data from Google Cloud Storage
    raw_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/raw/fake-news-classification.zip")
    processed_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/processed/preprocessed_data.csv")
    predict_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/predict/predict_data.csv")

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


def upload_to_gcs(file_obj, bucket_name, destination_blob_name):
    """Uploads a file object to a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    file_obj.seek(0)
    blob.upload_from_file(file_obj, content_type="application/octet-stream")
    print(f"Best model saved to GCS: gs://{bucket_name}/{destination_blob_name}")


def create_tmp_model_folder(cfg: DictConfig, local: bool, best_artifact):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download the best model to a temporary directory
        artifact_dir = best_artifact.download(root=tmp_dir)
        if local:
            tmp_dir = MODELS_DIR
        best_model_dir = os.path.join(tmp_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)

        for file_name in os.listdir(artifact_dir):
            full_file_name = os.path.join(artifact_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, os.path.join(best_model_dir, file_name))
        if local:
            print(f"Best model saved locally in: {best_model_dir}")
        else:
            with io.BytesIO() as model_bytes:
                for file_name in os.listdir(best_model_dir):
                    full_file_name = os.path.join(best_model_dir, file_name)
                    with open(full_file_name, "rb") as f:
                        model_bytes.write(f.read())
                model_bytes.seek(0)
                gcs_bucket_name = cfg.cloud.bucket_name_model
                gcs_model_path = os.path.join(cfg.cloud.model_dir, os.path.basename(best_model_dir) + ".ckpt")
                upload_to_gcs(model_bytes, gcs_bucket_name, gcs_model_path)


# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
