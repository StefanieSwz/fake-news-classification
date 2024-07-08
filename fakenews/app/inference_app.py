import os
import shutil
import tempfile
import pandas as pd
import hydra
from datetime import datetime
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest, REGISTRY, CONTENT_TYPE_LATEST
from starlette.responses import Response
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.model.model import BERTClass
from fakenews.config import add_to_database, download_model_from_gcs

app = FastAPI()

counter_requests = Counter("requests", "Number of requests made to FastAPI inference app.")


class Title(BaseModel):
    title: str


# Initialize global variables
model = None
device = None
cfg = None


@app.on_event("startup")
async def startup_event():
    global model, device, cfg

    # Load Hydra configuration
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Download the model from GCS
    bucket_name = cfg.cloud.bucket_name_model
    model_path = os.path.join(cfg.cloud.model_dir, "best_model.ckpt")
    local_model_path = os.path.join(tempfile.gettempdir(), "model.ckpt")
    download_model_from_gcs(bucket_name, model_path, local_model_path)

    # Load the trained model
    model = BERTClass.load_from_checkpoint(os.path.join(local_model_path), cfg=cfg)

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
    global model, device, cfg

    # Ensure model, device, and cfg are initialized
    if model is None or device is None or cfg is None:
        await startup_event()

    # Check if the file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Invalid file type. Please upload a CSV file.")

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

        # Check if the CSV file is empty
        if data.empty:
            raise HTTPException(status_code=422, detail="Uploaded CSV file is empty.")

        # Use parameters from the request or fallback to Hydra config
        batch_size = batch_size or cfg.train.batch_size
        max_length = max_length or cfg.preprocess.max_length

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
        background_tasks.add_task(add_to_database, cfg, db_predictions)
        return JSONResponse(result)


@app.post("/predict_single/")
async def predict_single(
    background_tasks: BackgroundTasks,
    title: Title,
    max_length: int = Query(default=None, description="Max length for preprocessing"),
):
    """Generate prediction for a single title."""
    global model, device, cfg

    # Ensure model, device, and cfg are initialized
    if model is None or device is None or cfg is None:
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
                cfg,
                [(now, title, prediction["predicted_label"], prediction["probability"])],
            )
            return JSONResponse(prediction)


Instrumentator().instrument(app).expose(app)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Fake News Inference API. Check /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))  # Ensure this port matches the Dockerfile EXPOSE port
    uvicorn.run(app, host="0.0.0.0", port=port)
