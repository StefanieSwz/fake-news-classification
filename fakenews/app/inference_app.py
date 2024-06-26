import os
import shutil
import tempfile
import pandas as pd
import hydra
from fastapi import FastAPI, UploadFile, File, Query
from starlette.responses import JSONResponse
import torch
import wandb
from fakenews.config import (
    MODEL_REGISTRY,
    WANDB_API_KEY,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, device, cfg

    # Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    # Load Hydra configuration
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Use the model artifact from wandb
    artifact = run.use_artifact(f"{WANDB_ENTITY}/model-registry/{MODEL_REGISTRY}:best", type="model")
    artifact_dir = artifact.download()

    # Load the trained model
    model = BERTClass.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), cfg=cfg)

    # Determine the device (CPU, GPU, MPS)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Move model to the device
    model.to(device)
    model.eval()


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    batch_size: int = Query(default=None, description="Batch size for prediction"),
    max_length: int = Query(default=None, description="Max length for preprocessing"),
):
    """Generate predictions for the uploaded CSV file."""
    global model, device, cfg

    # Use parameters from the request or fallback to Hydra config
    batch_size = batch_size or cfg.train.batch_size
    max_length = max_length or cfg.preprocess.max_length

    # Initialize the DataPreprocessor with the provided or default max_length
    preprocessor = DataPreprocessor(data_dir=None, max_length=max_length)

    # Save uploaded file to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load CSV file
        data = pd.read_csv(file_path)

        # Prepare DataLoader for prediction data
        predict_dataloader = preprocessor.create_prediction_dataloader_from_df(data, batch_size=batch_size)

        # Predict
        predictions = []
        with torch.no_grad():
            for batch in predict_dataloader:
                sent_id, mask = [t.to(device) for t in batch]
                outputs = model(sent_id=sent_id, mask=mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.tolist())

        # Convert predictions to a DataFrame and then to JSON
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        return JSONResponse(predictions_df.to_dict(orient="records"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
