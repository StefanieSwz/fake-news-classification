import torch.nn as nn
from transformers import AutoModel
from omegaconf import DictConfig
import hydra


class BERT_Class(nn.Module):
    """
    BERT-based classification model.

    Attributes:
        model_name (str): The name of the pretrained BERT model to use.
        dropout_rate (float): The dropout rate.
        hidden_size (int): The hidden size of the BERT model.
        intermediate_size (int): The size of the intermediate dense layer.
        output_size (int): The size of the output layer.
    """

    def __init__(
        self, model_name: str, dropout_rate: float, hidden_size: int, intermediate_size: int, output_size: int
    ):
        """
        Initializes the BERT_Class with a specified pretrained model.

        Args:
            model_name (str): The name of the pretrained BERT model to use.
            dropout_rate (float): The dropout rate.
            hidden_size (int): The hidden size of the BERT model.
            intermediate_size (int): The size of the intermediate dense layer.
            output_size (int): The size of the output layer.
        """
        super(BERT_Class, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        """
        Defines the forward pass of the model.

        Args:
            sent_id (torch.Tensor): The input IDs of the sentences.
            mask (torch.Tensor): The attention masks.

        Returns:
            torch.Tensor: The output logits of the model.
        """
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


@hydra.main(config_path="../../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
    Main function to initialize and print the model details.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    model = BERT_Class(
        model_name=cfg.model.name,
        dropout_rate=cfg.model.dropout_rate,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        output_size=cfg.model.output_size,
    )
    print(f"Using pretrained model: {cfg.model.name}")
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
