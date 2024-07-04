# Base image
FROM python:3.11-slim

# Install Python
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc curl gnupg apt-transport-https && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash && \
    /root/google-cloud-sdk/install.sh && \
    /root/google-cloud-sdk/bin/gcloud components install gsutil

# Add Google Cloud SDK to PATH
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"
# Copy service account key (if needed)
#COPY service-account-credentials.json /root/.gcloud/service-account-credentials.json

# Set environment variable to point to the key
#ENV GOOGLE_APPLICATION_CREDENTIALS="/root/.gcloud/service-account-credentials.json"
#RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY fakenews/ fakenews/
#COPY data/ data/
COPY config/ config/
#COPY .env /.env

WORKDIR /

#RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
#RUN pip install google-cloud-storage
RUN pip install -r requirements.txt --no-cache-dir

# Install DVC and Google Cloud Storage dependencies
# RUN pip install dvc[gs] google-cloud-storage google-cloud-secret-manager

# Download data from Google Cloud Storage bucket
#RUN gsutil cp -r gs://mlops-lmu-data-bucket/data /data/

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/fakenews"

# Set entrypoint
ENTRYPOINT ["python", "-u", "fakenews/model/train_model.py"]
