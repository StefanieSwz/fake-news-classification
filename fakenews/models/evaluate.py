import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from fakenews.models.model import BERT_Class
from fakenews.data.preprocessing import DataPreprocessor
from fakenews.config import PROCESSED_DATA_DIR, MODELS_DIR


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig):
    """
    Evaluate a trained BERT model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # Set device
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_checkpoint_path = os.path.join(to_absolute_path(MODELS_DIR), cfg.eval.model_checkpoint)
    print(f"Model checkpoint: {model_checkpoint_path}")

    # Initialize the model and load the checkpoint
    model = BERT_Class(
        model_name=cfg.model.name,
        dropout_rate=cfg.model.dropout_rate,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        output_size=cfg.model.output_size,
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))

    preprocessor = DataPreprocessor(to_absolute_path(PROCESSED_DATA_DIR), max_length=cfg.preprocess.max_length)
    _, _, test_dataloader = preprocessor.process(
        batch_size=cfg.eval.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=to_absolute_path(PROCESSED_DATA_DIR),
    )

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    criterion = nn.NLLLoss()

    # Evaluation loop
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            # Transfer batch to DEVICE
            sent_id, mask, labels = [item.to(DEVICE) for item in batch]
            preds = model(sent_id, mask)
            labels = labels.long()
            loss = criterion(preds, labels)
            total_loss += loss.item()

            # Collect predictions and labels
            preds = preds.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_test_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.3f}")
    print("Classification Report:")
    report = classification_report(all_labels, all_preds)
    print(report)

    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    accuracy = report_dict["accuracy"]
    print(f"Exact Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate()
