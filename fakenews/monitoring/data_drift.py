import pandas as pd
import datetime
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


def filter_dataframe(df, filter_value):
    """
    Filter a pandas DataFrame based on an integer or datetime value.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        filter_value (int or datetime): If an integer is provided, return the last n entries.
                                        If a datetime is provided, filter out observations earlier than this.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if isinstance(filter_value, int):
        # Return the last n entries
        return df.tail(filter_value)
    elif isinstance(filter_value, datetime):
        # Filter out observations earlier than the datetime
        return df[df["timestamp"] >= filter_value]
    else:
        raise ValueError("filter_value must be either an integer or a datetime object")


reference_data = pd.read_csv(PROCESSED_DATA_DIR / "preprocessed_data.csv")
reference_data = filter_dataframe(reference_data, filter_value=500)
current_data = pd.read_csv(MONITORING_DATA_DIR / "monitoring_db.csv")
current_data = filter_dataframe(current_data, filter_value=500)

reference_data.drop(reference_data.columns[0], axis=1, inplace=True)
current_data.drop(["timestamp", "probability"], axis=1, inplace=True)

reference_data = reference_data.astype({"title": str, "label": int})
current_data = current_data.astype({"title": str, "label": int})
reference_data["prediction"] = reference_data["label"]
current_data["prediction"] = current_data["label"]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# set either text=None or images=None when only the other is needed
inputs_reference = processor(
    text=reference_data["title"].tolist(),
    images=None,
    return_tensors="pt",
    padding=True,
    truncation=True,
)
text_features_reference = model.get_text_features(inputs_reference["input_ids"], inputs_reference["attention_mask"])
embeddings_reference = text_features_reference.detach().numpy()

emb_reference_df = pd.DataFrame(
    embeddings_reference, columns=[f"col{i+1}" for i in range(embeddings_reference.shape[1])]
)
reference_data.reset_index(drop=True, inplace=True)
# reference_data["embedding"] = text_features_reference.tolist()
reference_data = pd.concat(
    [
        reference_data,
        emb_reference_df,
    ],
    axis=1,
)

# set either text=None or images=None when only the other is needed
inputs_current = processor(
    text=current_data["title"].tolist(),
    images=None,
    return_tensors="pt",
    padding=True,
    truncation=True,
)

text_features_current = model.get_text_features(inputs_current["input_ids"], inputs_current["attention_mask"])
embeddings_current = text_features_current.detach().numpy()
emb_current_df = pd.DataFrame(embeddings_current, columns=[f"col{i+1}" for i in range(embeddings_current.shape[1])])
current_data.reset_index(drop=True, inplace=True)
# current_data["embedding"] = text_features_current.tolist()
current_data = pd.concat([current_data, emb_current_df], axis=1)

column_mapping = ColumnMapping()
column_mapping.target = "label"  #'y' is the name of the column with the target function
column_mapping.prediction = "prediction"
column_mapping.id = None  # there is no ID column in the dataset
column_mapping.embeddings = {"embedding": reference_data.columns[3:]}
column_mapping.text_features = ["title"]  #'title' is the name of the column with strings
column_mapping.target_names = {"0": "True", "1": "Fake"}
column_mapping.task = "classification"

# General Preset metrics
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
report_path = MONITORING_DIR / "data_drift_report.html"
report_general.save_html(str(report_path))

# general tests
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
data_test_path = MONITORING_DIR / "data_drift_tests.html"
data_test.save_html(str(data_test_path))

# text metrics
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
text_specific_metrics_report_path = MONITORING_DIR / "text_drift_metrics.html"
text_specific_metrics_report.save_html(str(text_specific_metrics_report_path))
