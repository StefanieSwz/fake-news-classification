# Makefile

Make commands we implemented to run parts of our code.
Benefits of using Makefile with hydra:

- **Modularity**: Independent execution of setup, installation, and training steps.
- **Consistency**: Uniform build and execution processes across environments.
- **Efficiency**: Saves time by avoiding full recompilation.
- **Reproducibility**: Simplifies the reproduction of experiments with consistent configurations.
- **Automation**: Reduces human error by automating repetitive and complex tasks.

## Create environment

```bash
make create_environment
conda activate fake-news-classification
```
## Data
### make data
Download data from Kaggle and save it in `data/raw/` folder (zipped or unzipped)

### make preprocess
Preprocesses the data and preprocessed data gets stored in `data/processed/`.

### make generate
Generates artificial predict data, by default 200 instances which can be changed by `cfg.predict.instances=300`. The generated data gets saved in `data/predict/`.

## Model

### make train

To train locally:
`make train ARGS="train.local_data=True train.local_wandb=True"`

To train without saving model in Wandb or GCS:
`make train ARGS="cloud.save_best_model_gcs=False train.log_model=False"`


Important configs:
```bash
    cloud.save_best_model_gcs=True  # whether to save model in GCS if val loss is lower
    train.log_model=True            # whether to save each trained model in wandb artifact registry
    train.sweep=False               # whether to do hyperparam sweep
    train.batch_size=32
    train.random_state=2018
    train.save_model=False          # whether to save model locally
    train.patience=3
    train.local_data=False          # whether to use local data or from GCS
    train.local_wandb=False         # whether to use local .env keys or from GCP
    train.pruning=False
    train.pruning_rate=0.3
    train.epochs=5
    train.devices=1
```

### make predict

Classifies `data/predict/predict_data.csv` as fake or real news. Both configs are by default set to False.

`make predict ARGS="predict.local_wandb=True predict.local_data=True"`

### make wandb_registry

Looks through Artifact registry and downloads best model and saves it in local `models/best_model/`. If models should be transferred to Model registry in Wandb, run

`make wandb_registry ARGS="predict.link_artifacts=True"`

Important configs:
```bash
    predict.link_artifacts=False        # whether to link models in Artifact registry to model registry
    predict.metric_name="val_loss"
    predict.save_best_model=True        # whether to download best model and save in local folder
    cloud.wandb_best_model_to_gcs=False # whether to save best model from model registry in GCS (overwrites current model)
```

### make train_distillation

Perform model distillation training by downloading the current best large model from the cloud and using it as the teacher model. The student model consists of a smaller
pretrained model with reduced finetuning layer size.

```bash
make train_distillation
```

All relevant configs of `make train` apply to this command as well, since it uses a similar training script. Further relvant configs are denoted in the `distillation.yaml`file:

```yaml
temperature: 2.0
alpha: 0.5
st_name_1: "albert-base-v2"
st_name_2: "distilbert-base-uncased"
st_hidden_size: 768  # Adjust based on the student model's hidden size
st_intermediate_size: 256  # Adjust as needed
class_name: "StudentBERTClass"
```

Here, the adjusted layers and pretrained models are denoted. The `class_name` denotes the student class used for training. For training a DistilBert student model, one may select `class_name=StudentDistilBertClass`. The parameters `temperature` and `alpha` denote the smoothing factor and the influence of the soft/hard loss on the overall distillation loss function.
The argument `cfg.cloud.distilled_file` determines the name the model is saved with in the model bucket on the cloud.

### make compare_models

This command downloads the current best large model and two trained student models from the Cloud Bucket and calculates FLOPS, inference time and test performance.

### make test_load

Activates load testing with locust. The locust page can be opened locally. Users and request interval can be selected to test the load on the FastAPI backend deployed in the Cloud Run for model inference.
