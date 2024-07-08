import pandas as pd
import hydra
from fastapi import FastAPI, HTTPException
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
    ClassificationPreset,
    TextOverviewPreset,
)
from evidently.metrics import (
    TextDescriptorsDriftMetric,
    TextDescriptorsDistribution,
    TextDescriptorsCorrelationMetric,
    EmbeddingsDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
    TestNumberOfEmptyRows,
    TestColumnsType,
    TestEmbeddingsDrift,
)
from transformers import CLIPProcessor, CLIPModel
from fakenews.config import setup_data_directories, upload_string_to_gcs
from datetime import datetime
import tempfile
from pathlib import Path

app = FastAPI()

# Initialize global variables
cfg = None
PROCESSED_DATA_DIR = None
MONITORING_DATA_DIR = None
model = None
processor = None
reference_data = None
current_data = None
column_mapping = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application, load configuration, and setup data directories and models."""
    global cfg, PROCESSED_DATA_DIR, MONITORING_DATA_DIR, model, processor, reference_data, current_data, column_mapping

    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    _, PROCESSED_DATA_DIR, _, MONITORING_DATA_DIR = setup_data_directories(cfg=cfg)
    PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)
    MONITORING_DATA_DIR = Path(MONITORING_DATA_DIR)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    reference_data, current_data = prepare_data(50)
    reference_data = get_embeddings(reference_data, processor, model)
    current_data = get_embeddings(current_data, processor, model)

    column_mapping = create_column_mapping(reference_data)


def filter_dataframe(df, filter_value):
    """
    Filter a pandas DataFrame based on an integer or datetime value.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        filter_value (int or datetime): The filter value.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if isinstance(filter_value, int):
        return df.tail(filter_value)
    elif isinstance(filter_value, datetime):
        return df[df["timestamp"] >= filter_value]
    else:
        raise ValueError("filter_value must be either an integer or a datetime object")


def prepare_data(filter_value: int):
    """
    Prepare the reference and current datasets for processing.

    Args:
        filter_value (int): The filter value for selecting data.

    Returns:
        tuple: A tuple containing the prepared reference and current data.
    """
    reference_data = pd.read_csv(PROCESSED_DATA_DIR / "preprocessed_data.csv")
    reference_data = filter_dataframe(reference_data, filter_value=filter_value)
    current_data = pd.read_csv(MONITORING_DATA_DIR / "monitoring_db.csv")
    current_data = filter_dataframe(current_data, filter_value=filter_value)

    reference_data.drop(reference_data.columns[0], axis=1, inplace=True)
    current_data.drop(["timestamp", "probability"], axis=1, inplace=True)

    reference_data = reference_data.astype({"title": str, "label": int})
    current_data = current_data.astype({"title": str, "label": int})
    reference_data["prediction"] = reference_data["label"]
    current_data["prediction"] = current_data["label"]

    return reference_data, current_data


def get_embeddings(data, processor, model):
    """
    Get embeddings for the text data.

    Args:
        data (pd.DataFrame): The data to process.
        processor (CLIPProcessor): The processor for data preprocessing.
        model (CLIPModel): The model to generate embeddings.

    Returns:
        pd.DataFrame: The data with embeddings added.
    """
    inputs = processor(
        text=data["title"].tolist(),
        images=None,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_features = model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    embeddings = text_features.detach().numpy()
    emb_df = pd.DataFrame(embeddings, columns=[f"col{i+1}" for i in range(embeddings.shape[1])])
    data.reset_index(drop=True, inplace=True)
    return pd.concat([data, emb_df], axis=1)


def create_column_mapping(reference_data):
    """
    Create a ColumnMapping object for Evidently reports.

    Args:
        reference_data (pd.DataFrame): The reference data to map.

    Returns:
        ColumnMapping: The column mapping configuration.
    """
    column_mapping = ColumnMapping()
    column_mapping.target = "label"
    column_mapping.prediction = "prediction"
    column_mapping.id = None
    column_mapping.embeddings = {"embedding": reference_data.columns[3:]}
    column_mapping.text_features = ["title"]
    column_mapping.target_names = {"0": "True", "1": "Fake"}
    column_mapping.task = "classification"
    return column_mapping


def generate_and_upload_report(reference_data, current_data, column_mapping, metrics, file_name):
    """
    Generate and upload an Evidently report.

    Args:
        reference_data (pd.DataFrame): The reference data.
        current_data (pd.DataFrame): The current data.
        column_mapping (ColumnMapping): The column mapping configuration.
        metrics (list): The list of metrics for the report.
        file_name (str): The destination file name in GCS.

    Returns:
        None
    """
    report = Report(metrics=metrics)
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        report.save_html(tmp.name)
        with open(tmp.name, "r") as file:
            report_html = file.read()
    upload_string_to_gcs(report_html, cfg.cloud.bucket_name_data, file_name, content_type="text/html")


def generate_and_upload_test_suite(reference_data, current_data, column_mapping, tests, file_name):
    """
    Generate and upload an Evidently test suite.

    Args:
        reference_data (pd.DataFrame): The reference data.
        current_data (pd.DataFrame): The current data.
        column_mapping (ColumnMapping): The column mapping configuration.
        tests (list): The list of tests for the suite.
        file_name (str): The destination file name in GCS.

    Returns:
        None
    """
    test_suite = TestSuite(tests=tests)
    test_suite.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        test_suite.save_html(tmp.name)
        with open(tmp.name, "r") as file:
            test_suite_html = file.read()
    upload_string_to_gcs(test_suite_html, cfg.cloud.bucket_name_data, file_name, content_type="text/html")


@app.get("/generate_data_drift_report/")
async def generate_data_drift_report(filter_value: int = 50):
    """
    Generate and upload a data drift report.

    Args:
        filter_value (int): The filter value for selecting data.

    Returns:
        dict: A success message.
    """
    try:
        global reference_data, current_data, column_mapping

        generate_and_upload_report(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
            metrics=[
                DataQualityPreset(),
                DataDriftPreset(),
                TargetDriftPreset(),
                ClassificationPreset(),
                TextOverviewPreset(column_name="title"),
            ],
            file_name="reports/monitoring/data_drift_report.html",
        )

        return {"message": "Data drift report generated and uploaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate_data_drift_tests/")
async def generate_data_drift_tests(filter_value: int = 50):
    """
    Generate and upload a data drift test suite.

    Args:
        filter_value (int): The filter value for selecting data.

    Returns:
        dict: A success message.
    """
    try:
        global reference_data, current_data, column_mapping

        generate_and_upload_test_suite(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
            tests=[
                TestNumberOfEmptyRows(),
                TestColumnsType(),
                TestShareOfDriftedColumns(),
                TestNumberOfDriftedColumns(),
                TestEmbeddingsDrift(embeddings_name="embedding"),
            ],
            file_name="reports/monitoring/data_drift_tests.html",
        )

        return {"message": "Data drift tests generated and uploaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate_text_drift_metrics/")
async def generate_text_drift_metrics(filter_value: int = 50):
    """
    Generate and upload a text drift metrics report.

    Args:
        filter_value (int): The filter value for selecting data.

    Returns:
        dict: A success message.
    """
    try:
        global reference_data, current_data, column_mapping

        generate_and_upload_report(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
            metrics=[
                TextDescriptorsDistribution(column_name="title"),
                TextDescriptorsDriftMetric(column_name="title"),
                TextDescriptorsCorrelationMetric(column_name="title"),
                EmbeddingsDriftMetric("embedding"),
            ],
            file_name="reports/monitoring/text_drift_metrics.html",
        )

        return {"message": "Text drift metrics report generated and uploaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Fake News Monitoring API. Check /docs for API documentation."}
