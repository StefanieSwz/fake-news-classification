# Inference
For deployment we wrapped our model into application using FastAPI. We first deployed the model locally. Afterwards we deployed it in the cloud, using Cloud Run.

## Local
Local deployment using FastAPI as backend and streamlit as frontend.
```bash
uvicorn fakenews.app.inference_app:app --reload # BACKEND
streamlit run fakenews/app/frontend.py          # FRONTEND
```
*bekomme da einen read timed out error (Anne).*

Curl commands can also be used for requesting the API. However, one may deem the frontend as more convenient.
An example for requesting the inference app deployed locally on port 8000:

For Mac:
```bash
curl -X POST "http://localhost:8000/predict_single/" \
     -H "Content-Type: application/json" \
     -d '{"title": "Example news title"}'
```
For Windows: `curl -X POST "http://localhost:8000/predict_single/" -H "Content-Type: application/json" -d "{\"title\": \"Example news title\"}"`

## Cloud Run

### Deploy model
We have the docker files and cloudbuild YAMLs for the backend and frontend application. Both get automatically deployed in Google Cloud's `Cloud Run` when pushing to main. This works with this command:

`gcloud builds submit --config=config/cloudbuild_<name>.yaml`

The command to deploy a model from local would be:

`gcloud run deploy backend --image=europe-west3-docker.pkg.dev/mlops-fakenews/mlops-fakenews-container/backend:latest --platform managed --region europe-west3 --allow-unauthenticated`

### Use deployed model

Important URLs of deployed models:

- [Frontend](https://frontend-awan6kp5bq-ey.a.run.app/docs)
- [Backend](https://backend-awan6kp5bq-ey.a.run.app/docs)
- [Monitoring](https://monitoring-awan6kp5bq-ey.a.run.app/docs)


How to use a curl request to backend deployed in the Cloud:

`curl -X POST "https://backend-awan6kp5bq-ey.a.run.app/predict_single/" -H "Content-Type: application/json" -d "{\"title\": \"Example news title\"}"`
