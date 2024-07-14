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
