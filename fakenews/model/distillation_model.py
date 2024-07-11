import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from omegaconf import DictConfig
from sklearn.metrics import classification_report


class StudentBERTClass(pl.LightningModule):
    """
    A BERT-based student model for text classification using knowledge distillation.

    Attributes:
        bert (nn.Module): Pretrained BERT model.
        dropout (nn.Dropout): Dropout layer.
        relu (nn.ReLU): ReLU activation function.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        softmax (nn.LogSoftmax): Log softmax activation function.
        criterion (nn.NLLLoss): Negative log likelihood loss.
        lr (float): Learning rate.
        dropout_rate (float): Dropout rate.
        batch_size (int): Batch size.
        test_preds (list): List to store test predictions.
        test_labels (list): List to store test labels.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the StudentBERTClass with a specified configuration.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.
        """
        super(StudentBERTClass, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.distillation.st_name_1)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(cfg.distillation.st_hidden_size, cfg.distillation.st_intermediate_size)
        self.fc2 = nn.Linear(cfg.distillation.st_intermediate_size, cfg.model.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.lr = cfg.train.lr
        self.dropout_rate = cfg.model.dropout_rate
        self.batch_size = cfg.train.batch_size
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

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Defines a single step in the training loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        sent_id, mask, labels = batch
        outputs = self(sent_id, mask)
        loss = self.criterion(outputs, labels)
        acc = (labels == outputs.argmax(dim=-1)).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step in the validation loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing the validation loss and accuracy.
        """
        sent_id, mask, labels = batch
        outputs = self(sent_id, mask)
        loss = self.criterion(outputs, labels)
        acc = (labels == outputs.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """
        Defines a single step in the test loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing the test predictions, labels, loss, and accuracy.
        """
        sent_id, mask, labels = batch
        outputs = self(sent_id, mask)
        loss = self.criterion(outputs, labels)
        acc = (labels == outputs.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss)

        # Store predictions and labels for the entire test set
        self.test_preds.append(outputs)
        self.test_labels.append(labels)

        return {"preds": outputs, "labels": labels, "loss": loss, "acc": acc}

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


class StudentDistilBERTClass(pl.LightningModule):
    """
    A DistilBERT-based student model for text classification using knowledge distillation.

    Attributes:
        bert (nn.Module): Pretrained DistilBERT model.
        dropout (nn.Dropout): Dropout layer.
        relu (nn.ReLU): ReLU activation function.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        softmax (nn.LogSoftmax): Log softmax activation function.
        criterion (nn.NLLLoss): Negative log likelihood loss.
        lr (float): Learning rate.
        dropout_rate (float): Dropout rate.
        batch_size (int): Batch size.
        test_preds (list): List to store test predictions.
        test_labels (list): List to store test labels.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the StudentDistilBERTClass with a specified configuration.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.
        """
        super(StudentDistilBERTClass, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.distillation.st_name_2)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(cfg.distillation.st_hidden_size, cfg.distillation.st_intermediate_size)
        self.fc2 = nn.Linear(cfg.distillation.st_intermediate_size, cfg.model.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.lr = cfg.train.lr
        self.dropout_rate = cfg.model.dropout_rate
        self.batch_size = cfg.train.batch_size
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
        # DistilBERT doesn't have a pooler_output, use the last hidden state
        cls_hs = self.bert(sent_id, attention_mask=mask).last_hidden_state[:, 0, :]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Defines a single step in the training loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        sent_id, mask, labels = batch
        outputs = self(sent_id, mask)
        loss = self.criterion(outputs, labels)
        acc = (labels == outputs.argmax(dim=-1)).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step in the validation loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing the validation loss and accuracy.
        """
        sent_id, mask, labels = batch
        outputs = self(sent_id, mask)
        loss = self.criterion(outputs, labels)
        acc = (labels == outputs.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """
        Defines a single step in the test loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing the test predictions, labels, loss, and accuracy.
        """
        sent_id, mask, labels = batch
        outputs = self(sent_id, mask)
        loss = self.criterion(outputs, labels)
        acc = (labels == outputs.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss)

        # Store predictions and labels for the entire test set
        self.test_preds.append(outputs)
        self.test_labels.append(labels)

        return {"preds": outputs, "labels": labels, "loss": loss, "acc": acc}

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


class DistillationTrainer(pl.LightningModule):
    """
    A PyTorch Lightning module for training the student model using knowledge distillation.

    Attributes:
        teacher_model (pl.LightningModule): Pretrained teacher model.
        student_model (pl.LightningModule): Student model to be trained.
        T (float): Temperature for distillation.
        alpha (float): Weight for distillation loss.
        test_preds (list): List to store test predictions.
        test_labels (list): List to store test labels.
    """

    def __init__(self, teacher_model, student_model, cfg):
        """
        Initializes the DistillationTrainer with teacher and student models.

        Args:
            teacher_model (pl.LightningModule): Pretrained teacher model.
            student_model (pl.LightningModule): Student model to be trained.
            cfg (DictConfig): Configuration composed by Hydra.
        """
        super(DistillationTrainer, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.T = cfg.distillation.temperature
        self.alpha = cfg.distillation.alpha
        self.test_preds = []
        self.test_labels = []

    def training_step(self, batch, batch_idx):
        """
        Defines a single step in the training loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        sent_id, mask, labels = batch
        with torch.no_grad():
            teacher_outputs = self.teacher_model(sent_id, mask)

        student_outputs = self.student_model(sent_id, mask)
        loss = distillation_loss(student_outputs, labels, teacher_outputs, self.T, self.alpha)
        acc = (labels == student_outputs.argmax(dim=-1)).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step in the validation loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing the validation loss and accuracy.
        """
        sent_id, mask, labels = batch
        student_outputs = self.student_model(sent_id, mask)
        loss = self.student_model.criterion(student_outputs, labels)
        acc = (labels == student_outputs.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """
        Defines a single step in the test loop.

        Args:
            batch (tuple): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing the test predictions, labels, loss, and accuracy.
        """
        sent_id, mask, labels = batch
        student_outputs = self.student_model(sent_id, mask)
        loss = self.student_model.criterion(student_outputs, labels)
        acc = (labels == student_outputs.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss)

        # Store predictions and labels for the entire test set
        self.test_preds.append(student_outputs)
        self.test_labels.append(labels)

        return {"preds": student_outputs, "labels": labels, "loss": loss, "acc": acc}

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
        Configures the optimizer for the student model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.AdamW(self.student_model.parameters(), lr=self.student_model.lr)


def distillation_loss(student_outputs, labels, teacher_outputs, temperature, alpha):
    """
    Computes the distillation loss.

    Args:
        student_outputs (torch.Tensor): Student model predictions.
        labels (torch.Tensor): Ground truth labels.
        teacher_outputs (torch.Tensor): Teacher model predictions (logits).
        temperature (float): Temperature parameter.
        alpha (float): Weight for distillation loss.

    Returns:
        torch.Tensor: Loss value combining cross-entropy and distillation loss.
    """
    hard_loss = F.cross_entropy(student_outputs, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction="batchmean",
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
