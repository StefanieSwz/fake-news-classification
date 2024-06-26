import os
import torch
from ts.torch_handler.base_handler import BaseHandler
from fakenews.data.preprocessing import DataPreprocessor


class BERTHandler(BaseHandler):
    """
    Custom handler for serving a BERT-based model using TorchServe.

    This handler processes input data, performs inference using a pre-trained
    and fine-tuned BERT model, and returns the predictions.

    Attributes:
        manifest (dict): The model's manifest file.
        preprocessor (DataPreprocessor): Preprocessor for handling input data.
        model (torch.jit.ScriptModule): The traced model loaded from a file.
    """

    def __init__(self):
        """
        Initialize the handler.
        """
        super(BERTHandler, self).__init__()

    def initialize(self, ctx):
        """
        Initialize the handler, loading the model and preprocessor.

        Args:
            ctx (context): The context object containing system properties and manifest.
        """
        # Load manifest and properties from context
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # Print statements for debugging
        print(f"Model directory: {model_dir}")

        # Initialize DataPreprocessor
        self.preprocessor = DataPreprocessor(data_dir=None, max_length=15)

        # Load the traced model
        model_path = os.path.join(model_dir, "deployable_model.pt")
        print(f"Model path: {model_path}")
        self.model = torch.jit.load(model_path)
        self.model.eval()  # Set the model to evaluation mode

    def preprocess(self, data):
        """
        Preprocess the input data to create a DataLoader for prediction.

        Args:
            data (list): The input data, expected to be a CSV file.

        Returns:
            DataLoader: DataLoader object for the prediction data.
        """
        # Get the file path from the input data
        file_path = data[0].get("data") or data[0].get("body")

        # Preprocess the CSV file to a DataFrame
        df = self.preprocessor.preprocess_csv(file_path)

        # Create a DataLoader from the DataFrame
        dataloader = self.preprocessor.create_prediction_dataloader_from_df(df, batch_size=32)
        return dataloader

    def inference(self, data, *args, **kwargs):
        """
        Perform inference on the preprocessed data.

        Args:
            data (DataLoader): DataLoader object containing the preprocessed data.

        Returns:
            list: List of predictions.
        """
        all_preds = []
        with torch.no_grad():  # Disable gradient calculation
            for batch in data:
                # Move tensors to the same device as the model
                sent_id, mask = [t.to(self.model.device) for t in batch]

                # Perform forward pass
                outputs = self.model(sent_id=sent_id, mask=mask)

                # Get the predicted class
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.tolist())
        return all_preds

    def postprocess(self, data):
        """
        Postprocess the inference output to return a list of predictions.

        Args:
            data (list): List of predictions.

        Returns:
            list: List of dictionaries containing the predictions.
        """
        return [{"prediction": pred} for pred in data]


# Instantiate the handler
_service = BERTHandler()
