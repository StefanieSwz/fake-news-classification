import os
import tempfile
from fastapi.testclient import TestClient
from fakenews.app.inference_app import app  # Adjust the import according to your actual application structure

client = TestClient(app)


def test_predict():
    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(b"title\n")
        temp_file.write(b"Test title 1\n")
        temp_file.write(b"Test title 2\n")
        temp_file.write(b"Test title 3\n")
        temp_file_path = temp_file.name

    with open(temp_file_path, "rb") as f:
        files = {"file": (os.path.basename(temp_file_path), f, "text/csv")}
        response = client.post("/predict/", files=files, params={"batch_size": 32, "max_length": 15})

    os.remove(temp_file_path)  # Clean up the temporary file

    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    for prediction in predictions:
        assert "title" in prediction
        assert "prediction" in prediction
        assert "predicted_label" in prediction
        assert "probability" in prediction


def test_predict_single():
    title = {"title": "Test title for single prediction"}
    response = client.post("/predict_single/", json=title, params={"max_length": 15})

    assert response.status_code == 200
    prediction = response.json()
    assert "title" in prediction
    assert "prediction" in prediction
    assert "predicted_label" in prediction
    assert "probability" in prediction


if __name__ == "__main__":
    import pytest

    pytest.main()
