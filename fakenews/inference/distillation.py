import os
import hydra
import torch
from fakenews.config import DISTILLED_MODEL, PROCESSED_DATA_DIR, load_gc_model
from fakenews.model.distillation_model import StudentBERTClass, StudentDistilBERTClass
from fakenews.model.model import BERTClass
from fakenews.data.preprocessing import DataPreprocessor
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from fakenews.inference.quantitization import measure_inference_time
from transformers import BertTokenizer
from thop import profile


def evaluate_model(model, test_dataloader, device):
    """
    Evaluate the model on the test set and return the metrics.

    Args:
        model: The model to evaluate.
        test_dataloader: DataLoader for the test data.
        device: The device to run the evaluation on.

    Returns:
        dict: A dictionary containing accuracy, F1 score, and confusion matrix.
    """
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

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": conf_matrix}


def main():
    # Initialize Hydra and load configuration
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Determine the device (CPU, GPU, MPS)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Download and load the BERT teacher model
    bert_path = load_gc_model(cfg)
    bert_model = BERTClass(cfg)
    checkpoint = torch.load(bert_path, map_location=device)
    bert_model.load_state_dict(checkpoint["state_dict"])

    # Download and load the StudentDistilBERT model
    stu_dist_path = load_gc_model(cfg, model_filename="stu_dist_bert_model.ckpt")
    dist_model = StudentDistilBERTClass(cfg)
    checkpoint = torch.load(stu_dist_path, map_location=device)
    dist_model.load_state_dict(checkpoint["state_dict"])

    # Save StudentDistilBERT model locally
    torch.save({"state_dict": dist_model.state_dict()}, os.path.join(DISTILLED_MODEL, "stu_dist_bert_model.ckpt"))

    # Download and load the StudentBERT model
    stu_bert_path = load_gc_model(cfg, model_filename="stu_bert_model.ckpt")
    stu_model = StudentBERTClass(cfg)
    checkpoint = torch.load(stu_bert_path, map_location=device)
    stu_model.load_state_dict(checkpoint["state_dict"])

    # Save StudentBERT model locally
    torch.save({"state_dict": stu_model.state_dict()}, os.path.join(DISTILLED_MODEL, "stu_bert_model.ckpt"))

    # Load the preprocessed test data
    preprocessor = DataPreprocessor(data_dir=PROCESSED_DATA_DIR, max_length=cfg.preprocess.max_length)
    _, _, test_dataloader = preprocessor.process(
        batch_size=cfg.train.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=PROCESSED_DATA_DIR,
    )

    # Evaluate each model and print the metrics
    bert_metrics = evaluate_model(bert_model, test_dataloader, device)
    dist_model_metrics = evaluate_model(dist_model, test_dataloader, device)
    stu_bert_metrics = evaluate_model(stu_model, test_dataloader, device)

    print("BERT Model Metrics:")
    print(f"Accuracy: {bert_metrics['accuracy']}")
    print(f"F1 Score: {bert_metrics['f1_score']}")
    print(f"Confusion Matrix:\n{bert_metrics['confusion_matrix']}")

    print("\nStudent DistilBERT Model Metrics:")
    print(f"Accuracy: {dist_model_metrics['accuracy']}")
    print(f"F1 Score: {dist_model_metrics['f1_score']}")
    print(f"Confusion Matrix:\n{dist_model_metrics['confusion_matrix']}")

    print("\nStudent BERT Model Metrics:")
    print(f"Accuracy: {stu_bert_metrics['accuracy']}")
    print(f"F1 Score: {stu_bert_metrics['f1_score']}")
    print(f"Confusion Matrix:\n{stu_bert_metrics['confusion_matrix']}")

    # Measure inference time
    tokenizer = BertTokenizer.from_pretrained(cfg.model.name)
    dummy_input = tokenizer(
        "This is a dummy input", return_tensors="pt", max_length=15, padding="max_length", truncation=True
    )
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)

    bert_inference_time = measure_inference_time(bert_model, input_ids, attention_mask, device)
    dist_model_inference_time = measure_inference_time(dist_model, input_ids, attention_mask, device)
    stu_bert_inference_time = measure_inference_time(stu_model, input_ids, attention_mask, device)

    print("\nInference Time (ms):")
    print(f"BERT Model: {bert_inference_time * 1000:.2f} ms")
    print(f"Student DistilBERT Model: {dist_model_inference_time * 1000:.2f} ms")
    print(f"Student BERT Model: {stu_bert_inference_time * 1000:.2f} ms")

    # Calculate FLOPs and parameters
    bert_flops, bert_params = profile(bert_model, inputs=(input_ids, attention_mask))
    dist_model_flops, dist_model_params = profile(dist_model, inputs=(input_ids, attention_mask))
    stu_bert_flops, stu_bert_params = profile(stu_model, inputs=(input_ids, attention_mask))

    print("\nFLOPs and Parameters:")
    print(f"BERT Model - FLOPs: {bert_flops}, Params: {bert_params}")
    print(f"Student DistilBERT Model - FLOPs: {dist_model_flops}, Params: {dist_model_params}")
    print(f"Student BERT Model - FLOPs: {stu_bert_flops}, Params: {stu_bert_params}")


if __name__ == "__main__":
    main()
