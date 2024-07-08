# locustfile.py

import os
import tempfile
from locust import HttpUser, between, task
from fakenews.config import get_backend_url


class MyUser(HttpUser):
    """A Locust user class that defines tasks to be performed by the users."""

    wait_time = between(1, 2)
    host = get_backend_url()  # Get the backend URL from the config module

    @task
    def predict(self):
        """
        A task that simulates a user uploading a CSV file to the /predict/ endpoint.

        This task creates a temporary CSV file with sample data, uploads it to the
        /predict/ endpoint, and then cleans up the temporary file.
        """
        # Create a temporary CSV file with sample data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(b"title\n")
            temp_file.write(b"Test title 1\n")
            temp_file.write(b"Test title 2\n")
            temp_file.write(b"Test title 3\n")
            temp_file_path = temp_file.name

        # Open the temporary file and upload it to the /predict/ endpoint
        with open(temp_file_path, "rb") as f:
            files = {"file": (os.path.basename(temp_file_path), f, "text/csv")}
            self.client.post("/predict/", files=files, params={"batch_size": 32, "max_length": 15})

        # Clean up the temporary file
        os.remove(temp_file_path)

    @task
    def predict_single(self):
        """
        A task that simulates a user sending a single title to the /predict_single/ endpoint.

        This task sends a JSON object with a sample title to the /predict_single/ endpoint.
        """
        # Define a sample title for single prediction
        title = {"title": "Test title for single prediction"}

        # Send the sample title to the /predict_single/ endpoint
        self.client.post("/predict_single/", json=title, params={"max_length": 15})
