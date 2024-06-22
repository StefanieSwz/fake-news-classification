# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY fakenews/ fakenews/
COPY data/ data/
COPY config/ config/
COPY .env /.env

WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/fakenews"

# Set entrypoint
ENTRYPOINT ["python", "-u", "fakenews/model/predict.py"]