import requests
from fakenews.config import get_backend_url

# Get the backend URL from the deployed Cloud Run service
backend_url = get_backend_url(service_name="monitoring", url="MONITORING_URL")


def test_generate_data_drift_report():
    """
    Test the /generate_data_drift_report/ endpoint of the deployed backend API by verifying
    the response contains the expected success message.
    """
    response = requests.get(f"{backend_url}/generate_data_drift_report/", params={"filter_value": 50})
    assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
    result = response.json()
    assert "message" in result
    assert result["message"] == "Data drift report generated and uploaded successfully"


def test_generate_data_drift_tests():
    """
    Test the /generate_data_drift_tests/ endpoint of the deployed backend API by verifying
    the response contains the expected success message.
    """
    response = requests.get(f"{backend_url}/generate_data_drift_tests/", params={"filter_value": 50})
    assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
    result = response.json()
    assert "message" in result
    assert result["message"] == "Data drift tests generated and uploaded successfully"


def test_generate_text_drift_metrics():
    """
    Test the /generate_text_drift_metrics/ endpoint of the deployed backend API by verifying
    the response contains the expected success message.
    """
    response = requests.get(f"{backend_url}/generate_text_drift_metrics/", params={"filter_value": 50})
    assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
    result = response.json()
    assert "message" in result
    assert result["message"] == "Text drift metrics report generated and uploaded successfully"


def test_generate_data_drift_report_invalid_filter():
    """
    Test the /generate_data_drift_report/ endpoint with an invalid filter value.
    """
    response = requests.get(f"{backend_url}/generate_data_drift_report/", params={"filter_value": "invalid"})
    assert response.status_code == 422, f"Unexpected status code: {response.status_code} - {response.text}"


def test_generate_data_drift_tests_invalid_filter():
    """
    Test the /generate_data_drift_tests/ endpoint with an invalid filter value.
    """
    response = requests.get(f"{backend_url}/generate_data_drift_tests/", params={"filter_value": "invalid"})
    assert response.status_code == 422, f"Unexpected status code: {response.status_code} - {response.text}"


def test_generate_text_drift_metrics_invalid_filter():
    """
    Test the /generate_text_drift_metrics/ endpoint with an invalid filter value.
    """
    response = requests.get(f"{backend_url}/generate_text_drift_metrics/", params={"filter_value": "invalid"})
    assert response.status_code == 422, f"Unexpected status code: {response.status_code} - {response.text}"
