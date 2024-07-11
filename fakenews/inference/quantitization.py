import os
import torch
import time
from fakenews.config import MODELS_DIR, load_gc_model
from fakenews.model.model import BERTClass
import hydra
from transformers import BertTokenizer, AutoModel
from thop import profile
import multiprocessing


class PlainBERTModel(torch.nn.Module):
    """
    A plain PyTorch model class wrapping a pretrained BERT model with additional layers for fine-tuning.

    Args:
        model_name (str): The name of the pretrained BERT model.
        hidden_size (int): The hidden size of the BERT model.
        intermediate_size (int): The size of the intermediate layer.
        output_size (int): The size of the output layer.
        dropout_rate (float): The dropout rate.
    """

    def __init__(self, model_name, hidden_size, intermediate_size, output_size, dropout_rate):
        super(PlainBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        """
        Forward pass for the model.

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


def measure_inference_time(model, input_ids, attention_mask, device, num_runs=100):
    """
    Measure the average inference time of a model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_ids (torch.Tensor): The input IDs.
        attention_mask (torch.Tensor): The attention masks.
        device (torch.device): The device to run the model on.
        num_runs (int, optional): The number of runs to average. Defaults to 100.

    Returns:
        float: The average inference time.
    """
    model.to(device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_ids, attention_mask)
            times.append(time.time() - start_time)
    avg_time = sum(times) / num_runs
    return avg_time


def main():
    """
    Main function to load the model, quantize it, and measure inference times.
    """
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        cfg = hydra.compose(config_name="config")

    # Download the model from GCS
    local_model_path = load_gc_model(cfg)
    pl_model = BERTClass.load_from_checkpoint(os.path.join(local_model_path), cfg=cfg)

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
    print(plain_model)

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

    # Measure inference time for the original model
    original_inference_time = measure_inference_time(plain_model, input_ids, attention_mask, device)
    print(f"Original Model - Average Inference Time: {original_inference_time}")

    # Quantization
    torch.backends.quantized.engine = "qnnpack"
    quantized_model = torch.quantization.quantize_dynamic(plain_model, {torch.nn.Linear}, dtype=torch.qint8)

    quantized_model_path = os.path.join(MODELS_DIR, "quantized_model.pth")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

    # Load the quantized model
    state_dict = torch.load(quantized_model_path)
    quantized_model.load_state_dict(state_dict)

    # Measure inference time for the quantized model
    quantized_inference_time = measure_inference_time(quantized_model, input_ids, attention_mask, device)
    print(f"Quantized Model - Average Inference Time: {quantized_inference_time}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
