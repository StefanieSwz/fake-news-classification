# test_api.py
import os
import tempfile
import requests
from fakenews.config import get_backend_url

# Get the backend URL from the deployed Cloud Run service
backend_url = get_backend_url()


def test_predict():
    """
    Test the /predict/ endpoint of the deployed backend API by uploading a CSV file
    and verifying the response contains the expected predictions.
    """
    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(b"title\n")
        temp_file.write(b"Test title 1\n")
        temp_file.write(b"Test title 2\n")
        temp_file.write(b"Test title 3\n")
        temp_file_path = temp_file.name

    with open(temp_file_path, "rb") as f:
        files = {"file": (os.path.basename(temp_file_path), f, "text/csv")}
        response = requests.post(f"{backend_url}/predict/", files=files, params={"batch_size": 32, "max_length": 15})

    os.remove(temp_file_path)  # Clean up the temporary file

    assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    for prediction in predictions:
        assert "title" in prediction
        assert "prediction" in prediction
        assert "predicted_label" in prediction
        assert "probability" in prediction


def test_predict_single():
    """
    Test the /predict_single/ endpoint of the deployed backend API by sending a single title
    and verifying the response contains the expected prediction.
    """
    title = {"title": "Test title for single prediction"}
    response = requests.post(f"{backend_url}/predict_single/", json=title, params={"max_length": 15})

    assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
    prediction = response.json()
    assert "title" in prediction
    assert "prediction" in prediction
    assert "predicted_label" in prediction
    assert "probability" in prediction


def test_predict_empty_file():
    """
    Test the /predict/ endpoint with an empty CSV file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(b"title\n")
        temp_file_path = temp_file.name

    with open(temp_file_path, "rb") as f:
        files = {"file": (os.path.basename(temp_file_path), f, "text/csv")}
        response = requests.post(f"{backend_url}/predict/", files=files, params={"batch_size": 32, "max_length": 15})

    os.remove(temp_file_path)  # Clean up the temporary file

    assert response.status_code == 422, f"Unexpected status code: {response.status_code} - {response.text}"


def test_predict_invalid_file():
    """
    Test the /predict/ endpoint with an invalid file type.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is not a CSV file\n")
        temp_file_path = temp_file.name

    with open(temp_file_path, "rb") as f:
        files = {"file": (os.path.basename(temp_file_path), f, "text/plain")}
        response = requests.post(f"{backend_url}/predict/", files=files, params={"batch_size": 32, "max_length": 15})

    os.remove(temp_file_path)  # Clean up the temporary file

    assert response.status_code == 422, f"Unexpected status code: {response.status_code} - {response.text}"


def test_predict_large_file():
    """
    Test the /predict/ endpoint with a very large CSV file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(b"title\n")
        for i in range(1000):  # Generate a large number of titles
            temp_file.write(f"Test title {i}\n".encode("utf-8"))
        temp_file_path = temp_file.name

    with open(temp_file_path, "rb") as f:
        files = {"file": (os.path.basename(temp_file_path), f, "text/csv")}
        response = requests.post(f"{backend_url}/predict/", files=files, params={"batch_size": 32, "max_length": 15})

    os.remove(temp_file_path)  # Clean up the temporary file

    assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) == 1000
    for prediction in predictions:
        assert "title" in prediction
        assert "prediction" in prediction
        assert "predicted_label" in prediction
        assert "probability" in prediction
