import click
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AdamW
from fakenews.config import PROCESSED_DATA_DIR
from model import BERT_Class
from fakenews.data.preprocessing import DataPreprocessor

# Set device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

@click.group()
def cli():
    """Command line interface for training and evaluating the BERT model."""
    pass

@click.command()
@click.option("--pretrained", default="base", type=click.Choice(["base", "large"]), help="Pretrained model to use.")
@click.option("--lr", default=1e-5, help="Learning rate to use for training.")
@click.option("--batch_size", default=32, help="Batch size to use for training.")
@click.option("--epochs", default=10, help="Number of epochs to train for.")
@click.option("--patience", default=3, help="Epochs to wait for stopping early.")
def train(pretrained, epochs=10, lr=1e-5, batch_size=32, patience=3):
    """
    Train the BERT model.

    Args:
        pretrained (str): Pretrained model to use. Options are "base" and "large".
        epochs (int): Number of epochs to train for.
        lr (float): Learning rate to use for training.
        batch_size (int): Batch size to use for training.
        patience (int): Number of epochs to wait for improvement before stopping early.
    """
    print("Training day and night")
    print(f"{pretrained=}, {lr=}, {batch_size=}, {epochs=}, {patience=}")

    # Initialize the model and move it to the specified device
    model = BERT_Class(pretrained=pretrained).to(DEVICE)
    preprocessor = DataPreprocessor(PROCESSED_DATA_DIR)
    train_dataloader, val_dataloader, _ = preprocessor.process(batch_size)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    best_valid_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1} / {epochs}")

        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                print(f"  Batch {step}  of  {len(train_dataloader)}.")

            # Transfer batch to DEVICE
            sent_id, mask, labels = [item.to(DEVICE) for item in batch]

            # Zero gradients, perform forward pass, compute loss, and backpropagate
            model.zero_grad()
            preds = model(sent_id, mask)
            labels = labels.long()
            loss = criterion(preds, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"\nTraining Loss: {avg_train_loss:.3f}")

        model.eval()
        total_loss = 0

        # Validation loop
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if step % 50 == 0 and not step == 0:
                    print(f"  Batch {step}  of  {len(val_dataloader)}.")

                # Transfer batch to DEVICE
                sent_id, mask, labels = [item.to(DEVICE) for item in batch]
                preds = model(sent_id, mask)
                labels = labels.long()
                loss = criterion(preds, labels)
                total_loss += loss.item()

        avg_valid_loss = total_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_valid_loss:.3f}")

        # Early stopping logic
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_weights.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    torch.save(model.state_dict(), "final_model_weights.pt")

@click.command()
@click.argument("model_checkpoint")
@click.option("--pretrained", default="base", type=click.Choice(["base", "large"]), help="Pretrained model to use.")
@click.option("--batch_size", default=32, help="Batch size to use for evaluation.")
def evaluate(model_checkpoint, pretrained, batch_size=32):
    """
    Evaluate a trained model.

    Args:
        model_checkpoint (str): Path to the model checkpoint file.
        pretrained (str): Pretrained model to use. Options are "base" and "large".
        batch_size (int): Batch size to use for evaluation.
    """
    print("Evaluating like my life depends on it")
    print(f"Model checkpoint: {model_checkpoint}")

    # Initialize the model and load the checkpoint
    model = BERT_Class(pretrained=pretrained).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    preprocessor = DataPreprocessor(PROCESSED_DATA_DIR)
    _, _, test_dataloader = preprocessor.process(batch_size)

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

# Add commands to the CLI
cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
