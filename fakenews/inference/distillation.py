import hydra
from fakenews.config import load_gc_model
from fakenews.model.distillation_model import StudentDistilBERTClass

# Initialize Hydra and load configuration
with hydra.initialize(config_path="../../config", version_base="1.2"):
    cfg = hydra.compose(config_name="config")

# Specify the model filename to download
model_filename = "distilled_model.ckpt"  # or set it directly as 'distilled_model.ckpt'

# Download the model from GCS
local_model_path = load_gc_model(cfg, model_filename)

# Load the model checkpoint
pl_model = StudentDistilBERTClass.load_from_checkpoint(local_model_path, cfg=cfg)

# Ensure model is in evaluation mode
pl_model.eval()
