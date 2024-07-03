import os
import hydra
from omegaconf import DictConfig
import torch
import wandb
import tempfile

from fakenews.config import MODEL_REGISTRY, access_secret_version, setup_data_directories
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def predict(cfg: DictConfig):
    """
    Predict the labels for new instances using the trained model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """

    if cfg.predict.local_wandb:
        from fakenews.config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
    else:
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

    if cfg.predict.local_data:
        from fakenews.config import PREDICT_DATA_DIR
    else:
        PREDICT_DATA_DIR, _, _ = setup_data_directories()

    # Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    # Use the model artifact from wandb
    artifact = run.use_artifact(f"{WANDB_ENTITY}/model-registry/{MODEL_REGISTRY}:best", type="model")

    # Download the artifact to a temporary directory
    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact.download(root=artifact_dir)

        # Initialize the DataPreprocessor
        preprocessor = DataPreprocessor(data_dir=None, max_length=cfg.preprocess.max_length)

        # Create DataLoader for prediction data
        predict_dataloader = preprocessor.create_prediction_dataloader(
            PREDICT_DATA_DIR, batch_size=cfg.train.batch_size
        )

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
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(sent_id=input_ids, mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(zip(preprocessor.titles, preds.tolist(), probs.tolist()))

        # Convert predictions to a DataFrame and then to JSON
        result = []
        for title, pred, prob in predictions:
            result.append(
                {
                    "title": title,
                    "prediction": "real" if pred == 0 else "fake",
                    "predicted_label": pred,
                    "probability": prob[1] if pred == 1 else prob[0],
                }
            )
        return print(result)


if __name__ == "__main__":
    predict()
