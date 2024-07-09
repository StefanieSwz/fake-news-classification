import os
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModel
from fakenews.model.model import BERTClass
from fakenews.config import BEST_MODEL, DEPLOY_MODEL


class PlainBERTModel(torch.nn.Module):
    """
    A plain PyTorch implementation of a BERT-based model with additional layers.

    This model adds a dropout layer, a ReLU activation, and two fully connected layers on top of a pre-trained BERT model.

    Attributes:
        bert (AutoModel): The pre-trained BERT model.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        relu (torch.nn.ReLU): ReLU activation function.
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        softmax (torch.nn.LogSoftmax): LogSoftmax activation function.
    """

    def __init__(self, model_name, hidden_size, intermediate_size, output_size, dropout_rate):
        """
        Initialize the PlainBERTModel.

        Args:
            model_name (str): Name of the pre-trained BERT model.
            hidden_size (int): Size of the hidden layer.
            intermediate_size (int): Size of the intermediate layer.
            output_size (int): Size of the output layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(PlainBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        """
        Forward pass for the model.

        Args:
            sent_id (torch.Tensor): Tensor containing the token IDs.
            mask (torch.Tensor): Tensor containing the attention masks.

        Returns:
            torch.Tensor: The output predictions.
        """
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function to transform and save the model.

    This function loads a trained Lightning model, converts it to a plain PyTorch model, and traces it for deployment.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # Load the trained model with the specified configuration
    best_model_path = os.path.join(BEST_MODEL, "model.ckpt")
    lightning_model = BERTClass.load_from_checkpoint(best_model_path, cfg=cfg)

    # Create a plain PyTorch model
    plain_model = PlainBERTModel(
        model_name=cfg.model.name,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        output_size=cfg.model.output_size,
        dropout_rate=cfg.model.dropout_rate,
    )

    # Load the state dict from the lightning model into the plain model
    plain_model.load_state_dict(lightning_model.state_dict())

    # Create example inputs for tracing
    example_input = (torch.randint(0, 1000, (1, 128)), torch.ones(1, 128))

    # Trace the model
    traced_model = torch.jit.trace(plain_model, example_input)

    # Save the traced model
    deploy_model_path = os.path.join(DEPLOY_MODEL, "deployable_model.pt")
    os.makedirs(DEPLOY_MODEL, exist_ok=True)
    traced_model.save(deploy_model_path)
    print(f"Model successfully transformed and saved at '{deploy_model_path}'")


if __name__ == "__main__":
    main()
