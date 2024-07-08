# docker/backend.dockerfile
FROM python:3.11-slim

EXPOSE 8080

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc curl gnupg apt-transport-https && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash && \
    /root/google-cloud-sdk/install.sh && \
    /root/google-cloud-sdk/bin/gcloud components install gsutil

# Add Google Cloud SDK to PATH
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

COPY requirements.txt requirements.txt
COPY fakenews/ fakenews/
COPY config/ config/

WORKDIR /

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/fakenews"

# Set Google Application Credentials
ENV PORT 8080
CMD ["uvicorn", "fakenews.app.inference_app:app", "--host", "0.0.0.0", "--port", "8080"]
