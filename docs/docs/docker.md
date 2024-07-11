# Docker
Make sure to have [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) installed. The data also needs to be accessed, either with your local data folder or via the data bucket in GCS.

## About Docker
A Docker image is a template for a Docker container. A Docker container is a running instance of a Docker image. A Docker image is a static file, while a Docker container is a running process. Running inside a Docker container gives you a consistent and independent environment for your application. This means that you can be sure that your application will run the same way on your machine as it will on another machine. Thus, Docker gives the ability to abstract away the differences between different machines.

## General useful commands
```bash
docker images               # list all images
docker rmi <image_id>       # remove a docker image
docker run                  # create a docker container
docker start <container_id> # restart docker container
docker stop <container_id>  # stop docker container
docker ps                   # list containers
docker rm <container_id>    # remove docker container
```

## Local Docker
To run a dockerfile locally, for example to train the model, type
```bash
# Training
docker build -f docker/trainer_local.dockerfile . -t trainer:latest # creates a docker image
## Create Docker container with mounted models folder
docker run --name <container_name> -v %cd%/models:/models/ trainer:latest train.local_data=True train.local_wandb=True <model.dropout_rate=0.3 train.batch_size=8> # Windows command line
docker run --name <container_name> -v $(pwd)/models:/models/ trainer:latest # for Powershell/Linux
docker run --name <container_name> -v ${PWD}/models:/models/ trainer:latest # if something else does not work

# Predicting
docker build -f docker/predict_local.dockerfile . -t predict:latest
docker run --name <container_name> predict:latest predict.local_data=True predict.local_wandb=True
```

## Cloud Docker

Authentification for Docker with GCP: `gcloud auth configure-docker europe-west3.pkg.dev`

### Images
We use Cloud Build for building the containers in the cloud and Artifact registry for storing the images afterward. The latest image is automatically triggered when pushing code to main. We have 5 images in our Google Cloud [Artifact Registry](https://console.cloud.google.com/artifacts/docker/mlops-fakenews/europe-west3/mlops-fakenews-container?project=mlops-fakenews), `backend, frontend, monitoring, predict` and `trainer`.

### Cloudbuilds
To build containers in the cloud from local, we use the `cloudbuild.yaml` which in our case builds the docker images of `trainer` and `predict` and pushes them to the Artifact registry.

```bash
gcloud builds submit —-config=config/cloudbuild.yaml .            # for trainer and predict
gcloud builds submit —-config=config/cloudbuild_backend.yaml .    # builds, pushes and deploys
gcloud builds submit —-config=config/cloudbuild_frontend.yaml .   # builds, pushes and deploys
gcloud builds submit —-config=config/cloudbuild_monitoring.yaml . # builds, pushes and deploys
```

### Push und pull
To push and pull to our Artifact registry, we refer to this URL: `europe-west3-docker.pkg.dev/mlops-fakenews/mlops-fakenews-container`, e.g. to pull the latest trainer image, type

`docker pull europe-west3-docker.pkg.dev/mlops-fakenews/mlops-fakenews-container/trainer:latest`

Manually creating images & push them, example with busybox:
```bash
docker tag busybox <region>-docker.pkg.dev/<project-id>/<registry-name>/busybox:latest
docker push <region>-docker.pkg.dev/<project-id>/<registry-name>/busybox:latest
```
