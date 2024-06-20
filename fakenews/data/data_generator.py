import pandas as pd
import os
import hydra
from omegaconf import DictConfig
from fakenews.config import PROCESSED_DATA_DIR, PREDICT_DATA_DIR

@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def generate_instances(cfg: DictConfig):
    """
    Generate new instances for prediction based on the original preprocessed data.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # Load the original preprocessed data
    original_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "preprocessed_data.csv"), index_col=0)

    # Generate new instances based on the original data
    new_instances = []
    for _ in range(cfg.predict.instances):
        sample = original_data.sample(n=1, replace=True)
        new_instances.append({
            "title": sample["title"].values[0],
            "label": sample["label"].values[0]
        })

    # Create a DataFrame for the new instances
    new_instances_df = pd.DataFrame(new_instances)

    # Ensure the row indices are preserved
    new_instances_df.index.name = ""

    # Save the new instances to predict.csv
    predict_file_path = os.path.join(PREDICT_DATA_DIR, "predict_data.csv")
    if not os.path.exists(PREDICT_DATA_DIR):
        os.makedirs(PREDICT_DATA_DIR)
    new_instances_df.to_csv(predict_file_path, index=True)

if __name__ == "__main__":
    generate_instances()
