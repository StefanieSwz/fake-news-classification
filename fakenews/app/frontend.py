import requests
import streamlit as st
from fakenews.config import get_backend_url


def classify_csv(file, backend_url, batch_size):
    """
    Send the CSV file to the backend for classification.

    Args:
        file (UploadFile): The uploaded CSV file containing titles to classify.
        backend_url (str): The URL of the FastAPI backend.
        batch_size (int): The batch size for processing.

    Returns:
        dict or None: The JSON response from the backend containing predictions,
                      or None if the request failed.
    """
    predict_url = f"{backend_url}/predict/"
    params = {"batch_size": batch_size}
    response = requests.post(predict_url, params=params, files={"file": file}, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def classify_single(title, backend_url):
    """
    Send a single title to the backend for classification.

    Args:
        title (str): The title to classify.
        backend_url (str): The URL of the FastAPI backend.

    Returns:
        dict or None: The JSON response from the backend containing the prediction,
                      or None if the request failed.
    """
    predict_url = f"{backend_url}/predict_single/"
    response = requests.post(predict_url, json={"title": title}, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def main():
    """
    Main function of the Streamlit frontend.

    Provides an interface to upload a CSV file or enter a single title for classification
    using a FastAPI backend.
    """
    backend = get_backend_url()
    if backend is None:
        raise ValueError("Backend service not found")
    st.title("Fake News Classification")

    st.header("Classify a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Add inputs for batch_size
    batch_size = st.number_input("Batch Size", min_value=1, value=32)

    if uploaded_file is not None:
        result = classify_csv(uploaded_file, backend_url=backend, batch_size=batch_size)

        if result is not None:
            # Display the prediction results
            for item in result:
                st.write(f"Title: {item['title']}")
                st.write(f"Prediction: {item['prediction']}")
                st.write(f"Predicted Label: {item['predicted_label']}")
                st.write(f"Probability: {item['probability']}")
                st.write("---")
        else:
            st.write("Failed to get prediction")

    st.header("Classify a Single Title")
    single_title = st.text_input("Enter a title")

    if st.button("Classify Title"):
        if single_title:
            result = classify_single(single_title, backend_url=backend)

            if result is not None:
                st.write(f"Title: {result['title']}")
                st.write(f"Prediction: {result['prediction']}")
                st.write(f"Predicted Label: {result['predicted_label']}")
                st.write(f"Probability: {result['probability']}")
            else:
                st.write("Failed to get prediction")
        else:
            st.write("Please enter a title to classify.")


if __name__ == "__main__":
    main()
