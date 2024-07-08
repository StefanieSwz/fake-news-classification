import os
from pathlib import Path
import io

from dotenv import load_dotenv
import shutil

import pandas as pd

from omegaconf import DictConfig
from google.cloud import secretmanager, storage

import streamlit as st
from google.cloud import run_v2

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
MONITORING_DATA_DIR = DATA_DIR / "monitoring"

MODELS_DIR = PROJ_ROOT / "models"
BEST_MODEL = MODELS_DIR / "best_model"
DEPLOY_MODEL = MODELS_DIR / "deploy"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MONITORING_DIR = REPORTS_DIR / "monitoring"


def access_secret_version(secret_id):
    """
    Access the latest version of a secret from Secret Manager.

    This function retrieves the latest version of a secret stored in Google Cloud Secret Manager.
    It initializes a Secret Manager client, constructs the resource name for the secret version, and
    accesses the secret version to retrieve its payload.

    Args:
        secret_id (str): The identifier of the secret to access.

    Returns:
        str: The decoded payload of the secret.
    """
    client = secretmanager.SecretManagerServiceClient()
    project_id = "mlops-fakenews"  # Replace with your project ID
    secret_version_id = "latest"
    secret_version_name = f"projects/{project_id}/secrets/{secret_id}/versions/{secret_version_id}"
    response = client.access_secret_version(name=secret_version_name)
    return response.payload.data.decode("UTF-8")


def get_blob_from_gcs(bucket_name, blob_name):
    """
    Fetch a blob from Google Cloud Storage and return it as bytes.

    This function initializes a Google Cloud Storage client, accesses the specified bucket,
    retrieves the specified blob, and downloads its content as bytes.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        blob_name (str): The name of the blob to fetch from the bucket.

    Returns:
        bytes: The content of the blob as bytes.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def setup_data_directories(cfg: DictConfig):
    """
    Fetch data from Google Cloud Storage and set up temporary directories for data.

    This function creates temporary directories for raw, processed, and prediction data. It then
    fetches the data from Google Cloud Storage, saves it to the temporary directories, and returns
    the paths to these directories.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.

    Returns:
        tuple: A tuple containing paths to the temporary directories:
            - predict_data_dir (str): Path to the temporary directory for prediction data.
            - processed_data_dir (str): Path to the temporary directory for processed data.
            - raw_data_dir (str): Path to the temporary directory for raw data.
            - monitoring_data_dir (str): Path to the temporary directory for monitoring data.
    """
    # Create temporary directories
    raw_data_dir = tempfile.mkdtemp()
    processed_data_dir = tempfile.mkdtemp()
    predict_data_dir = tempfile.mkdtemp()
    monitoring_data_dir = tempfile.mkdtemp()

    # Access raw and processed data from Google Cloud Storage
    raw_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/raw/fake-news-classification.zip")
    processed_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/processed/preprocessed_data.csv")
    predict_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/predict/predict_data.csv")
    monitoring_data = get_blob_from_gcs(cfg.cloud.bucket_name_data, "data/monitoring/monitoring_db.csv")

    # Save raw and processed data to temporary files
    raw_data_path = os.path.join(raw_data_dir, "fake-news-classification.zip")
    processed_data_path = os.path.join(processed_data_dir, "preprocessed_data.csv")
    predict_data_path = os.path.join(predict_data_dir, "predict_data.csv")
    monitoring_data_path = os.path.join(monitoring_data_dir, "monitoring_db.csv")

    with open(raw_data_path, "wb") as f:
        f.write(raw_data)

    with open(processed_data_path, "wb") as f:
        f.write(processed_data)

    with open(predict_data_path, "wb") as f:
        f.write(predict_data)

    with open(monitoring_data_path, "wb") as f:
        f.write(monitoring_data)

    return predict_data_dir, processed_data_dir, raw_data_dir, monitoring_data_dir


def upload_to_gcs(file_obj, bucket_name, destination_blob_name):
    """
    Upload a file object to a Google Cloud Storage (GCS) bucket.

    This function uploads the given file object to the specified GCS bucket under the specified blob name.

    Args:
        file_obj (file-like object): The file object to be uploaded. The file pointer should be at the beginning of the file.
        bucket_name (str): The name of the GCS bucket to upload the file to.
        destination_blob_name (str): The destination path and name of the blob in the GCS bucket.

    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    file_obj.seek(0)
    blob.upload_from_file(file_obj, content_type="application/octet-stream")
    print(f"Best model saved to GCS: gs://{bucket_name}/{destination_blob_name}")


def create_tmp_model_folder(cfg: DictConfig, local: bool, best_artifact):
    """
    Create a temporary model folder and handle model storage.

    This function creates a temporary directory to store the best model downloaded from a given artifact.
    Depending on the `local` flag, it either saves the model locally or uploads it to Google Cloud Storage (GCS).

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        local (bool): Flag indicating whether to save the model locally or upload to GCS.
        best_artifact: The artifact object representing the best model to be downloaded.

    Returns:
        None
    """
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


def upload_string_to_gcs(content, bucket_name, destination_blob_name, content_type="text/csv"):
    """
    Upload a string to a Google Cloud Storage (GCS) bucket.

    This function uploads the given string to the specified GCS bucket under the specified blob name.

    Args:
        content (str): The file object to be uploaded. The file pointer should be at the beginning of the file.
        bucket_name (str): The name of the GCS bucket to upload the file to.
        destination_blob_name (str): The destination path and name of the blob in the GCS bucket.
        content_type (str): The MIME type of the content to be uploaded. Defaults to 'text/csv'.

    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(content, content_type=content_type)
    print(f"Uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")


def get_string_from_gcs(bucket_name, blob_name):
    """
    Fetch a blob from Google Cloud Storage and return it as a string.

    This function initializes a Google Cloud Storage client, accesses the specified bucket,
    retrieves the specified blob, and downloads its content as a string.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        blob_name (str): The name of the blob to fetch from the bucket.

    Returns:
        str: The content of the blob as a string.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def compare_and_upload_best_model(cfg: DictConfig, model_checkpoint_path, val_loss):
    """
    Compare the model's validation loss with the best model in GCS and upload if it's better.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra.
        model_checkpoint_path (str): Path to the saved model checkpoint.
        val_loss (float): Validation loss of the current model.

    Returns:
        None
    """
    print("Comparing model to best model in GCS")
    best_val_loss_cloud = float("inf")
    try:
        best_val_loss_cloud = float(
            get_string_from_gcs(
                cfg.cloud.bucket_name_model, os.path.join(cfg.cloud.val_loss_dir, cfg.cloud.val_loss_file)
            )
        )
    except Exception as e:
        print(f"Error fetching best validation loss: {e}\n Setting best_val_loss_cloud to infinity.")

    if val_loss < best_val_loss_cloud:
        print("New best model found. Uploading to GCS.")
        with open(model_checkpoint_path, "rb") as model_file:
            upload_to_gcs(
                model_file, cfg.cloud.bucket_name_model, os.path.join(cfg.cloud.model_dir, cfg.cloud.model_file)
            )
        upload_string_to_gcs(
            str(val_loss),
            cfg.cloud.bucket_name_model,
            os.path.join(cfg.cloud.val_loss_dir, cfg.cloud.val_loss_file),
        )
    else:
        print("Model not better than best model in GCS. Not uploading.")


def download_model_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """
    Downloads a blob from a Google Cloud Storage bucket and saves it to a local file.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob in the GCS bucket.
        destination_file_name (str): The local file path where the blob will be saved.

    Returns:
        None

    Example:
        download_model_from_gcs('my-bucket', 'path/to/blob', 'local/path/to/save')
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def add_to_database(cfg: DictConfig, predictions: list) -> None:
    """
    Uploads a list of predictions to a Google Cloud Storage bucket, appending to the existing CSV file.

    This function downloads the existing CSV file from GCS, appends the new predictions, and re-uploads the combined data.

    Args:
        cfg (DictConfig): Configuration object composed by Hydra, containing cloud settings.
        predictions (list): List of predictions to be added. Each prediction is a tuple of (timestamp, title, label, probability).

    Returns:
        None
    """
    file_path = "data/monitoring/monitoring_db.csv"
    bucket_name = cfg.cloud.bucket_name_data
    blob_name = file_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the existing CSV file from GCS if it exists
    if blob.exists():
        existing_data = blob.download_as_text()
        df_existing = pd.read_csv(io.StringIO(existing_data))
        print("Existing data downloaded from GCS.")
    else:
        # Create an empty DataFrame if the CSV file does not exist
        df_existing = pd.DataFrame(columns=["timestamp", "title", "label", "probability"])
        print("No existing data found. Creating a new DataFrame.")

    # Create a DataFrame for the new data
    df_new = pd.DataFrame(predictions, columns=["timestamp", "title", "label", "probability"])

    # Append the new data to the existing data
    df_combined = pd.concat([df_existing, df_new])

    # Convert combined data to CSV string
    csv_data = df_combined.to_csv(index=False, encoding="utf-8")
    print("CSV data prepared for upload.")

    # Upload the updated CSV string to GCS
    upload_string_to_gcs(csv_data, bucket_name, blob_name)


@st.cache_resource
def get_backend_url(service_name="backend", url="BACKEND_URL"):
    """Get the URL of the backend service."""
    parent = "projects/mlops-fakenews/locations/europe-west3"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == service_name:
            return service.uri
    return os.getenv(url, None)
