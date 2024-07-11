# Model training

We can train the model locally or in the Cloud. By default, if a model has a lower validation loss than the best model stored in GCS, it will be stored in GCS as best model. If a model should not be automatically be replaced by a newly trained model, set `cfg.cloud.save_best_model_gcs=False`.

## Local training
To train from local, we can make use of our Makefile and change the default hydra configuration as well. Training with defaults can be run with `make train`. Models are by default logged to wandb but not automatically saved to a local folder. Logging and model saving can be deactivated during training with the following commands:

`make train ARGS="train.save_model=False train.log_model=False”`

The default parameters in the `train.yaml`are set to the following:

```bash
lr: 1e-5
batch_size: 32
epochs: 1
patience: 3
random_state: 2018
test_size: 0.3
val_size: 0.5
filename: 'model'
verbose: True
devices: 1
log_every_n_steps: 50
refresh_rate: 50
precision: "32"
profiler: "simple"
num_runs: 5
sweep: False
save_model: False
log_model: True
local_wandb: False
local_data: False
```
To train with hyperparameter optimization, sweep can be run with `make train ARGS=train.sweep=True`.


## Training on Google Cloud

We used two different services to train on the Cloud, Vertex AI and the Compute Engine. We were able to increase quota for the Compute Engine such that we can train on a Nvidia T4 GPU. For Vertex AI, we did not get a quota increase, i.e. we only train on a CPU.

### Vertex AI

1. Check that `config_cpu.yaml` is in `config/`. It specifies the machine type we are using (`n1-highnem-2`) and the docker image saved in the Artifact registry `europe-west3-docker.pkg.dev/mlops-fakenews/mlops-fakenews-container/trainer:latest`.
2. Train in the cloud by `gcloud ai custom-jobs create --region=europe-west3 --display-name=train-run --config=config/config_cpu.yaml --service-account=sa-mlops@mlops-fakenews.iam.gserviceaccount.com` optionally specify parameters: `--args train.epochs=1`

3. Monitor on GCP: `Vertex AI > Training > Custom Jobs > View Logs`

### Compute Engine

To train on a GPU, we selected an image with Nvidia drivers and Pytorch (`c0-deeplearning-common-cu121-v20240627-debian-11-py310`) to create a Virtual Machine with 100GB storage and `n1-standard-8` machine type. When starting The VM for the first time you are prompted: `install nvidia driver?[y/n]` and then type `y`. Git is already installed, `nvidia-smi` and `conda`are available.

The rest of the training is analoug to local training.
