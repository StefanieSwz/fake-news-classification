from omegaconf import DictConfig
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AutoModel


class BERTClass(pl.LightningModule):
    """
    BERT-based classification model using PyTorch Lightning.

    Attributes:
        model_name (str): The name of the pretrained BERT model to use.
        dropout_rate (float): The dropout rate.
        hidden_size (int): The hidden size of the BERT model.
        intermediate_size (int): The size of the intermediate dense layer.
        output_size (int): The size of the output layer.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the BERTClass with a specified pretrained model and configuration.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.
        """
        super(BERTClass, self).__init__()
        self.save_hyperparameters(cfg)
        self.bert = AutoModel.from_pretrained(cfg.model.name)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(cfg.model.hidden_size, cfg.model.intermediate_size)
        self.fc2 = nn.Linear(cfg.model.intermediate_size, cfg.model.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.test_preds = []
        self.test_labels = []

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

    def training_step(self, batch, batch_idx):
        """
        Defines a single step in the training loop.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.

        Returns:
            The computed loss for the batch.
        """
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.hparams.train.lr)
        self.log("dropout_rate", self.hparams.model.dropout_rate)
        self.log("batch_size", self.hparams.train.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step in the validation loop.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.

        Returns:
            The computed loss for the batch.
        """
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """
        Defines a single step in the test loop.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.

        Returns:
            The computed loss for the batch.
        """
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("test_loss", loss)

        # Store predictions and labels for the entire test set
        self.test_preds.append(preds)
        self.test_labels.append(batch[2])  # batch[2] contains the labels

        return {"preds": preds, "labels": batch[2], "loss": loss, "acc": acc}

    def on_test_epoch_end(self):
        """
        Computes metrics at the end of the test epoch.
        """
        preds = torch.cat(self.test_preds).argmax(dim=1)  # Convert preds to discrete labels
        labels = torch.cat(self.test_labels)
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), output_dict=True)
        self.log("test_accuracy", report["accuracy"])
        print("Classification Report:")
        print(report)

        # Clear the lists for the next test
        self.test_preds.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            The configured optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.train.lr)


    def _get_preds_loss_accuracy(self, batch):
        """Convenience function since train/valid/test steps are similar."""
        sent_id, mask, labels = batch
        preds = self(sent_id, mask)
        loss = self.criterion(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        return preds, loss, acc
