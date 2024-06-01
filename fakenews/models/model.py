import torch.nn as nn
from transformers import AutoModel

class BERT_Class(nn.Module):
    PRETRAINED_MODELS = {
        "base": "bert-base-uncased",
        "large": "bert-large-uncased"
    }

    def __init__(self, pretrained="base"):
        """
        Initializes the BERT_Class with a specified pretrained model.

        Args:
            pretrained (str): The key for the pretrained model to use (default is "base").
                              Options are "base" for "bert-base-uncased" and "large" for "bert-large-uncased".
        """
        super(BERT_Class, self).__init__()
        model_name = self.PRETRAINED_MODELS.get(pretrained, "bert-base-uncased")
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # relu activation function
        self.fc1 = nn.Linear(768 if pretrained == "base" else 1024, 512)  # dense layer 1
        self.fc2 = nn.Linear(512, 2)  # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax activation function

    def forward(self, sent_id, mask):  # define the forward pass
        """
        Defines the forward pass of the model.

        Args:
            sent_id (torch.Tensor): The input IDs of the sentences.
            mask (torch.Tensor): The attention masks.

        Returns:
            torch.Tensor: The output logits of the model.
        """
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        # pass the inputs to the model
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer
        x = self.softmax(x)  # apply softmax activation
        return x

if __name__ == "__main__":
    # Example usage with predefined pretrained models
    base_model = BERT_Class(pretrained="base")
    print(f"Using pretrained model: base (bert-base-uncased)")
    print(f"Model architecture: {base_model}")
    print(f"Number of parameters: {sum(p.numel() for p in base_model.parameters())}")

    # Another example with a different model
    large_model = BERT_Class(pretrained="large")
    print(f"Using pretrained model: large (bert-large-uncased)")
    print(f"Model architecture: {large_model}")
    print(f"Number of parameters: {sum(p.numel() for p in large_model.parameters())}")
