import os
import torch
from fakenews.config import BEST_MODEL, PROCESSED_DATA_DIR, MODELS_DIR
from fakenews.model.model import BERTClass
from fakenews.data.preprocessing import DataPreprocessor
import hydra
from transformers import BertTokenizer
from transformers import AutoModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from thop import profile


class PlainBERTModel(torch.nn.Module):
    def __init__(self, model_name, hidden_size, intermediate_size, output_size, dropout_rate):
        super(PlainBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


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

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, conf_matrix


def main():
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Load the model checkpoint
    model_checkpoint_path = os.path.join(BEST_MODEL, "model" + ".ckpt")
    pl_model = BERTClass.load_from_checkpoint(model_checkpoint_path, cfg=cfg)

    # Create a plain PyTorch model
    plain_model = PlainBERTModel(
        model_name=cfg.model.name,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        output_size=cfg.model.output_size,
        dropout_rate=cfg.model.dropout_rate,
    )

    # Load the state dict from the lightning model into the plain model
    plain_model.load_state_dict(pl_model.state_dict())
    plain_model.to("cpu")
    device = torch.device("cpu")

    # Calculate FLOPs
    tokenizer = BertTokenizer.from_pretrained(cfg.model.name)
    dummy_input = tokenizer(
        "This is a dummy input", return_tensors="pt", max_length=15, padding="max_length", truncation=True
    )
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)

    # Calculate FLOPs for the original model
    flops, params = profile(plain_model, inputs=(input_ids, attention_mask))
    print(f"Original Model - FLOPs: {flops}, Params: {params}")

    # Quantization
    torch.backends.quantized.engine = "qnnpack"

    quantized_model = torch.quantization.quantize_dynamic(plain_model, {torch.nn.Linear}, dtype=torch.qint8)

    print("Quantized model:", quantized_model)

    quantized_model_path = os.path.join(MODELS_DIR, "quantized_model.pth")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

    # Load the quantized model
    state_dict = torch.load(quantized_model_path)
    quantized_model.load_state_dict(state_dict)

    # Calculate FLOPs for the quantized model
    flops, params = profile(quantized_model, inputs=(input_ids, attention_mask))
    print(f"Quantized Model - FLOPs: {flops}, Params: {params}")

    # Evaluate the quantized model
    quantized_model.to(device)

    preprocessor = DataPreprocessor(data_dir=PROCESSED_DATA_DIR, max_length=cfg.preprocess.max_length)
    _, _, test_dataloader = preprocessor.process(
        batch_size=cfg.train.batch_size,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.train.random_state,
        processed_data_dir=PROCESSED_DATA_DIR,
    )

    quantized_accuracy, quantized_f1, quantized_conf_matrix = evaluate_model(quantized_model, test_dataloader, device)
    print(f"Quantized Model - Accuracy: {quantized_accuracy}")
    print(f"Quantized Model - F1 Score: {quantized_f1}")
    print(f"Quantized Model - Confusion Matrix:\n{quantized_conf_matrix}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
