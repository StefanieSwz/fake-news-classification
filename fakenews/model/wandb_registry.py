import os
import operator
import shutil
import tempfile
import hydra
from omegaconf import DictConfig
import wandb

from fakenews.config import MODEL_REGISTRY, MODELS_DIR, WANDB_ENTITY, WANDB_PROJECT, WANDB_API_KEY


def link_all_artifacts_to_registry(cfg: DictConfig):
    """
    Link all artifacts to the model registry.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    wandb.login(key=WANDB_API_KEY)
    api = wandb.Api()

    # Retrieve all artifacts of type 'model'
    runs = api.runs(path=f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    for run in runs:
        try:
            for artifact in run.logged_artifacts():
                if artifact.type == "model":
                    artifact.link(target_path=f"{WANDB_ENTITY}/model-registry/{MODEL_REGISTRY}", aliases=["latest"])
                    print(f"Model linked to registry: {artifact.name}")
        except wandb.errors.CommError as e:
            print(f"Error accessing artifacts for run {run.id}: {e}")


def stage_best_model_to_registry(cfg: DictConfig):
    """
    Stage the best model to the model registry and save it locally.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    metric_name = cfg.predict.metric_name
    higher_is_better = cfg.predict.higher_is_better

    api = wandb.Api()

    # Retrieve all artifacts of type 'model'
    runs = api.runs(path=f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None

    for run in runs:
        try:
            for artifact in run.logged_artifacts():
                if artifact.type == "model":
                    print(f"Checking artifact: {artifact.name}")
                    if metric_name in artifact.metadata:
                        metric_value = artifact.metadata[metric_name]
                        print(f"Found metric '{metric_name}' with value: {metric_value}")
                        if compare_op(metric_value, best_metric):
                            best_metric = metric_value
                            best_artifact = artifact
                    else:
                        print(f"Metric '{metric_name}' not found in artifact metadata.")
        except wandb.errors.CommError as e:
            print(f"Error accessing artifacts for run {run.id}: {e}")
        except Exception as e:
            print(f"Unexpected error accessing artifacts for run {run.id}: {e}")

    if best_artifact is None:
        print("No model found in registry.")
        return

    print(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    try:
        best_artifact.link(target_path=f"{WANDB_ENTITY}/model-registry/{MODEL_REGISTRY}", aliases=["best", "staging"])
        print("Model staged to registry.")

        if cfg.predict.save_best_model:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download the best model to a temporary directory
                artifact_dir = best_artifact.download(root=tmp_dir)
                best_model_dir = os.path.join(MODELS_DIR, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)

                for file_name in os.listdir(artifact_dir):
                    full_file_name = os.path.join(artifact_dir, file_name)
                    if os.path.isfile(full_file_name):
                        shutil.move(full_file_name, os.path.join(best_model_dir, file_name))

            print(f"Best model saved locally in: {best_model_dir}")

    except wandb.errors.CommError as e:
        print(f"Error linking artifact: {e}")


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.predict.link_artifacts:
        link_all_artifacts_to_registry(cfg)
    stage_best_model_to_registry(cfg)


if __name__ == "__main__":
    main()
