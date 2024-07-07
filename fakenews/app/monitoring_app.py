import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
from fakenews.config import PROCESSED_DATA_DIR, MONITORING_DATA_DIR, MONITORING_DIR
from fakenews.monitoring.data_drift import filter_dataframe

app = FastAPI()


@app.get("/run_analysis/")
async def run_analysis(filter_value: int = 50):
    try:
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

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        inputs_reference = processor(
            text=reference_data["title"].tolist(),
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_features_reference = model.get_text_features(
            inputs_reference["input_ids"], inputs_reference["attention_mask"]
        )
        embeddings_reference = text_features_reference.detach().numpy()
        emb_reference_df = pd.DataFrame(
            embeddings_reference,
            columns=[f"col{i+1}" for i in range(embeddings_reference.shape[1])],
        )
        reference_data.reset_index(drop=True, inplace=True)
        reference_data = pd.concat(
            [
                reference_data,
                emb_reference_df,
            ],
            axis=1,
        )

        inputs_current = processor(
            text=current_data["title"].tolist(),
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_features_current = model.get_text_features(inputs_current["input_ids"], inputs_current["attention_mask"])
        embeddings_current = text_features_current.detach().numpy()
        emb_current_df = pd.DataFrame(
            embeddings_current, columns=[f"col{i+1}" for i in range(embeddings_current.shape[1])]
        )
        current_data.reset_index(drop=True, inplace=True)
        current_data = pd.concat([current_data, emb_current_df], axis=1)

        column_mapping = ColumnMapping()
        column_mapping.target = "label"
        column_mapping.prediction = "prediction"
        column_mapping.id = None
        column_mapping.embeddings = {"embedding": reference_data.columns[3:]}
        column_mapping.text_features = ["title"]
        column_mapping.target_names = {"0": "True", "1": "Fake"}
        column_mapping.task = "classification"

        report_general = Report(
            metrics=[
                DataQualityPreset(),
                DataDriftPreset(),
                TargetDriftPreset(),
                ClassificationPreset(),
                TextOverviewPreset(column_name="title"),
            ]
        )
        report_general.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        report_path = os.path.join(MONITORING_DIR, "data_drift_report.html")
        report_general.save_html(report_path)

        data_test = TestSuite(
            tests=[
                TestNumberOfEmptyRows(),
                TestColumnsType(),
                TestShareOfDriftedColumns(),
                TestNumberOfDriftedColumns(),
                TestEmbeddingsDrift(embeddings_name="embedding"),
            ]
        )
        data_test.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        data_test_path = os.path.join(MONITORING_DIR, "data_drift_tests.html")
        data_test.save_html(data_test_path)

        text_specific_metrics_report = Report(
            metrics=[
                TextDescriptorsDistribution(column_name="title"),
                TextDescriptorsDriftMetric(column_name="title"),
                TextDescriptorsCorrelationMetric(column_name="title"),
                EmbeddingsDriftMetric("embedding"),
            ]
        )
        text_specific_metrics_report.run(
            reference_data=reference_data, current_data=current_data, column_mapping=column_mapping
        )
        text_specific_metrics_report_path = os.path.join(MONITORING_DIR, "text_drift_metrics.html")
        text_specific_metrics_report.save_html(text_specific_metrics_report_path)

        return {"message": "Reports generated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_report/")
async def download_report(report_type: str):
    if report_type not in ["data_drift_report", "data_drift_tests", "text_drift_metrics"]:
        raise HTTPException(status_code=400, detail="Invalid report type")

    file_path = os.path.join(MONITORING_DIR, f"{report_type}.html")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(path=file_path, filename=f"{report_type}.html")


# To run the FastAPI app, use the following command:
# uvicorn script_name:app --reload
