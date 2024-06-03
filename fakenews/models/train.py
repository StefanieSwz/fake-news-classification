import os
import torch
import torch.nn as nn
from fakenews.models.model import BERT_Class
from fakenews.data.preprocessing import DataPreprocessor
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from fakenews.config import PROCESSED_DATA_DIR, MODELS_DIR


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """
    Train the BERT model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(
        f"Training with configuration: lr={cfg.train.lr}, batch_size={cfg.train.batch_size}, epochs={cfg.train.epochs}, patience={cfg.train.patience}"
    )

    model = BERT_Class(
        model_name=cfg.model.name,
        dropout_rate=cfg.model.dropout_rate,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        output_size=cfg.model.output_size,
    ).to(DEVICE)

    preprocessor = DataPreprocessor(PROCESSED_DATA_DIR, cfg.preprocess.max_length)
    train_dataloader, val_dataloader, _ = preprocessor.process(
        batch_size=cfg.train.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=PROCESSED_DATA_DIR,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    criterion = nn.NLLLoss()

    best_valid_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.train.epochs):
        print(f"\nEpoch {epoch + 1} / {cfg.train.epochs}")

        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            sent_id, mask, labels = [item.to(DEVICE) for item in batch]

            model.zero_grad()
            preds = model(sent_id, mask)
            labels = labels.long()
            loss = criterion(preds, labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0:
                print(f"  Batch {step}  of  {len(train_dataloader)} - Training Loss: {loss.item():.3f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\nAverage Training Loss: {avg_train_loss:.3f}")

        model.eval()
        total_valid_loss = 0

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                sent_id, mask, labels = [item.to(DEVICE) for item in batch]
                preds = model(sent_id, mask)
                labels = labels.long()
                loss = criterion(preds, labels)
                total_valid_loss += loss.item()

                if step % 50 == 0:
                    print(f"  Batch {step}  of  {len(val_dataloader)} - Validation Loss: {loss.item():.3f}")

        avg_valid_loss = total_valid_loss / len(val_dataloader)
        print(f"Average Validation Loss: {avg_valid_loss:.3f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model_weights.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve == cfg.train.patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "final_model_weights.pt"))


if __name__ == "__main__":
    train()
