import requests
import streamlit as st
from fakenews.config import get_backend_url


def classify_csv(file, backend_url, batch_size, max_length):
    """
    Send the CSV file to the backend for classification.

    Args:
        file (UploadFile): The uploaded CSV file containing titles to classify.
        backend_url (str): The URL of the FastAPI backend.
        batch_size (int): The batch size for processing.
        max_length (int): The maximum length for tokenization.

    Returns:
        dict or None: The JSON response from the backend containing predictions,
                      or None if the request failed.
    """
    predict_url = f"{backend_url}/predict/"
    params = {"batch_size": batch_size, "max_length": max_length}
    response = requests.post(predict_url, params=params, files={"file": file}, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def classify_single(title, backend_url, max_length):
    """
    Send a single title to the backend for classification.

    Args:
        title (str): The title to classify.
        backend_url (str): The URL of the FastAPI backend.
        max_length (int): The maximum length for tokenization.

    Returns:
        dict or None: The JSON response from the backend containing the prediction,
                      or None if the request failed.
    """
    predict_url = f"{backend_url}/predict_single/"
    params = {"max_length": max_length}
    response = requests.post(predict_url, params=params, json={"title": title}, timeout=10)
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

    # Add inputs for batch_size and max_length
    batch_size = st.number_input("Batch Size", min_value=1, value=32)
    max_length_csv = st.number_input("Max Length for CSV", min_value=1, value=25)

    if uploaded_file is not None:
        result = classify_csv(
            uploaded_file,
            backend_url=backend,
            batch_size=batch_size,
            max_length=max_length_csv,
        )

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

    # Add input for max_length specific to single title classification
    max_length_single = st.number_input("Max Length for Single Title", min_value=1, value=25)

    if st.button("Classify Title"):
        if single_title:
            result = classify_single(single_title, backend_url=backend, max_length=max_length_single)

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
