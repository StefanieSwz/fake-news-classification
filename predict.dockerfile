# Base image
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY fakenews/ fakenews/
COPY data/ data/
COPY config/ config/
COPY .env /.env

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir
#RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Install DVC and Google Cloud Storage dependencies
RUN pip install dvc[gs] google-cloud-storage

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/fakenews"

# Set entrypoint
ENTRYPOINT ["python", "-u", "fakenews/model/predict.py"]
