# docker/frontend.dockerfile
FROM python:3.11-slim

EXPOSE 8080

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc curl gnupg apt-transport-https && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements_frontend.txt requirements_frontend.txt
COPY fakenews/ fakenews/
COPY config/ config/
#COPY service_account_credentials.json /service_account_credentials.json

WORKDIR /

RUN pip install --upgrade pip && \
    pip install -r requirements_frontend.txt --no-cache-dir

# Set the Python path
ENV PYTHONPATH="${PYTHONPATH}:/fakenews"
ENV PORT 8080

CMD ["streamlit", "run", "fakenews/app/frontend.py", "--server.port", "8080"]
