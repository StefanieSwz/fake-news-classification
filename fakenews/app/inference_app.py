import os
import shutil
import tempfile
import pandas as pd
import hydra
import csv
from datetime import datetime
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import wandb
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest, REGISTRY, CONTENT_TYPE_LATEST
from starlette.responses import Response

from fakenews.config import (
    MODEL_REGISTRY,
    WANDB_API_KEY,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass


app = FastAPI()

counter_requests = Counter("requests", "Number of requests made to FastAPI inference app.")


class Title(BaseModel):
    title: str


# Initialize global variables
model = None
device = None
cfg = None
artifact_dir = None


def add_to_database(predictions: list):
    """Simple function to add a list of predictions to the database."""
    file_path = "data/monitoring/monitoring_db.csv"

    if not os.path.isfile(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(columns=["timestamp", "title", "label", "probability"])
        df.to_csv(file_path, index=False, encoding="utf-8")

    with open(file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        for now, title, label, probability in predictions:
            writer.writerow([now, title, label, probability])


@app.on_event("startup")
async def startup_event():
    global model, device, cfg, artifact_dir

    # Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    # Load Hydra configuration
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Use the model artifact from wandb
    artifact = run.use_artifact(f"{WANDB_ENTITY}/model-registry/{MODEL_REGISTRY}:best", type="model")

    # Download the artifact to a temporary directory
    artifact_dir = tempfile.TemporaryDirectory()
    artifact.download(root=artifact_dir.name)

    # Load the trained model
    model = BERTClass.load_from_checkpoint(os.path.join(artifact_dir.name, "model.ckpt"), cfg=cfg)

    # Determine the device (CPU, GPU, MPS)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Move model to the device
    model.to(device)
    model.eval()


@app.post("/predict/")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    batch_size: int = Query(default=None, description="Batch size for prediction"),
    max_length: int = Query(default=None, description="Max length for preprocessing"),
):
    """Generate predictions for the uploaded CSV file."""
    global model, device, cfg, artifact_dir

    # Ensure model, device, and cfg are initialized
    if model is None or device is None or cfg is None or artifact_dir is None:
        await startup_event()

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
        db_predictions = []
        with torch.no_grad():
            for batch in predict_dataloader:
                sent_id, mask = [t.to(device) for t in batch]
                outputs = model(sent_id=sent_id, mask=mask)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(zip(data["title"].tolist(), preds.tolist(), probs.tolist()))

        # Convert predictions to a DataFrame and then to JSON
        result = []
        now = str(datetime.now())
        for title, pred, prob in predictions:
            result.append(
                {
                    "title": title,
                    "prediction": "real" if pred == 0 else "fake",
                    "predicted_label": pred,
                    "probability": prob[1] if pred == 1 else prob[0],
                }
            )
            counter_requests.inc()  # Increment by 1
            db_predictions.append((now, str(title), pred, prob[1] if pred == 1 else prob[0]))
        background_tasks.add_task(add_to_database, db_predictions)
        return JSONResponse(result)


@app.post("/predict_single/")
async def predict_single(
    background_tasks: BackgroundTasks,
    title: Title,
    max_length: int = Query(default=None, description="Max length for preprocessing"),
):
    """Generate prediction for a single title."""
    global model, device, cfg, artifact_dir

    # Ensure model, device, and cfg are initialized
    if model is None or device is None or cfg is None or artifact_dir is None:
        await startup_event()

    max_length = max_length or cfg.preprocess.max_length

    # Initialize the DataPreprocessor with the default max_length from config
    preprocessor = DataPreprocessor(data_dir=None, max_length=max_length)

    # Transform the input JSON to a DataFrame with a title column
    data = pd.DataFrame([title.dict()])

    # Prepare DataLoader for the single title
    predict_dataloader = preprocessor.create_prediction_dataloader_from_df(data, batch_size=1)

    # Predict
    with torch.no_grad():
        for batch in predict_dataloader:
            sent_id, mask = [t.to(device) for t in batch]
            outputs = model(sent_id=sent_id, mask=mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, dim=1)
            prediction = {
                "title": title.title,
                "prediction": "real" if pred.item() == 0 else "fake",
                "predicted_label": pred.item(),
                "probability": probs[0][1].item() if pred.item() == 1 else probs[0][0].item(),
            }
            counter_requests.inc()  # Increment by 1
            now = str(datetime.now())
            background_tasks.add_task(
                add_to_database,
                [(now, title, prediction["predicted_label"], prediction["probability"])],
            )
            return JSONResponse(prediction)


Instrumentator().instrument(app).expose(app)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
