import os
import csv
import torch
from fakenews.config import BEST_MODEL, MODELS_DIR, PROCESSED_DATA_DIR
from fakenews.model.model import BERTClass
from fakenews.data.preprocessing import DataPreprocessor
import hydra
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def read_csv_to_set(file_path):
    if os.path.exists(file_path):
        with open(file_path, mode="r") as infile:
            reader = csv.reader(infile)
            return {rows[0] for rows in reader if rows}
    else:
        return set()


def write_set_to_csv(keys_set, file_path):
    with open(file_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        for key in keys_set:
            writer.writerow([key])


def load_model_state(model, state_dict, missing_keys, unexpected_keys):
    # current_missing_keys, current_unexpected_keys = set(), set()
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        error_message = str(e)
        if "Missing key(s) in state_dict:" in error_message:
            missing_keys.update(
                set(
                    error_message.split("Missing key(s) in state_dict: ")[1]
                    .split("Unexpected key(s)")[0]
                    .strip()
                    .split(", ")
                )
            )
        if "Unexpected key(s) in state_dict:" in error_message:
            unexpected_keys.update(set(error_message.split("Unexpected key(s) in state_dict: ")[1].strip().split(", ")))

    return missing_keys, unexpected_keys


def evaluate_model(model, test_dataloader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            sent_id, mask, labels = batch
            sent_id, mask = sent_id.to(device), mask.to(device)
            outputs = model(sent_id=sent_id, mask=mask)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, conf_matrix


def main():
    # Load configuration
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Read missing and unexpected keys from CSV files
    missing_keys = read_csv_to_set("/Users/toby/fake-news-classification/fakenews/inference/missing_keys.csv")
    unexpected_keys = read_csv_to_set("/Users/toby/fake-news-classification/fakenews/inference/unexpected_keys.csv")

    # Load the model checkpoint
    model_checkpoint_path = os.path.join(BEST_MODEL, "model" + ".ckpt")
    pl_model = BERTClass.load_from_checkpoint(model_checkpoint_path, cfg=cfg)

    # Move model to CPU
    device = torch.device("cpu")
    pl_model.to(device)

    # Load the preprocessed test data
    preprocessor = DataPreprocessor(data_dir=PROCESSED_DATA_DIR, max_length=cfg.preprocess.max_length)
    _, _, test_dataloader = preprocessor.process(
        batch_size=cfg.train.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=PROCESSED_DATA_DIR,
    )

    # Evaluate original model
    original_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    original_accuracy, original_f1, original_conf_matrix = evaluate_model(pl_model, test_dataloader, original_device)
    print(f"Original Model - Accuracy: {original_accuracy}")
    print(f"Original Model - F1 Score: {original_f1}")
    print(f"Original Model - Confusion Matrix:\n{original_conf_matrix}")

    # Set the quantization backend
    torch.backends.quantized.engine = "qnnpack"

    # Perform dynamic quantization on the PyTorch model
    quantized_model = torch.quantization.quantize_dynamic(pl_model, {torch.nn.Linear}, dtype=torch.qint8)

    print("Quantized model:", quantized_model)

    # Save the quantized model state dict
    quantized_model_path = os.path.join(MODELS_DIR, "quantized_model.pth")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

    # Load the quantized model state dict
    state_dict = torch.load(quantized_model_path)

    # Initialize the model and load the state dict with error handling
    model = BERTClass(cfg)
    missing_keys, unexpected_keys = load_model_state(model, state_dict, missing_keys, unexpected_keys)

    # Save missing and unexpected keys to CSV files
    write_set_to_csv(missing_keys, "/Users/toby/fake-news-classification/fakenews/inference/missing_keys.csv")
    write_set_to_csv(unexpected_keys, "/Users/toby/fake-news-classification/fakenews/inference/unexpected_keys.csv")

    # Reload the model state dict after saving the keys
    state_dict = {k: v for k, v in state_dict.items() if k not in unexpected_keys}
    model_state_dict = model.state_dict()
    for k in missing_keys:
        if k in state_dict:
            model_state_dict[k] = state_dict[k]
    model.load_state_dict(model_state_dict)

    # Evaluate quantized model
    quantized_accuracy, quantized_f1, quantized_conf_matrix = evaluate_model(model, test_dataloader, original_device)
    print(f"Quantized Model - Accuracy: {quantized_accuracy}")
    print(f"Quantized Model - F1 Score: {quantized_f1}")
    print(f"Quantized Model - Confusion Matrix:\n{quantized_conf_matrix}")


# Ensure the main function is called correctly
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
