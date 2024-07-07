# docker/backend.dockerfile
FROM python:3.11-slim

EXPOSE 8080

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc curl gnupg apt-transport-https && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY fakenews/ fakenews/
COPY config/ config/
#COPY service_account_credentials.json /service_account_credentials.json

WORKDIR /

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/fakenews"

# Set Google Application Credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="service_account_credentials.json"
ENV PORT 8080
CMD ["uvicorn", "fakenews.app.inference_app:app", "--host", "0.0.0.0", "--port", "8080"]
