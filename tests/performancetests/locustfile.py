import os
import tempfile
from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)
    host = "http://localhost:8000"  # Set the host to your FastAPI application's URL

    @task
    def predict(self):
        """A task that simulates a user uploading a CSV file to the /predict/ endpoint."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(b"title\n")
            temp_file.write(b"Test title 1\n")
            temp_file.write(b"Test title 2\n")
            temp_file.write(b"Test title 3\n")
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as f:
            files = {"file": (os.path.basename(temp_file_path), f, "text/csv")}
            self.client.post("/predict/", files=files, params={"batch_size": 32, "max_length": 15})

        os.remove(temp_file_path)  # Clean up the temporary file

    @task
    def predict_single(self):
        """A task that simulates a user sending a single title to the /predict_single/ endpoint."""
        title = {"title": "Test title for single prediction"}
        self.client.post("/predict_single/", json=title, params={"max_length": 15})
