import operator
import hydra
from omegaconf import DictConfig
import wandb
import os

from fakenews.config import (
    MODEL_REGISTRY,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_API_KEY,
    create_tmp_model_folder,
    upload_string_to_gcs,
)


def link_all_artifacts_to_registry(cfg: DictConfig):
    """
    Link all artifacts to the model registry.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
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
            create_tmp_model_folder(cfg=cfg, local=True, best_artifact=best_artifact)

        if cfg.cloud.wandb_best_model_to_gcs:
            create_tmp_model_folder(cfg=cfg, local=False, best_artifact=best_artifact)
            # Upload validation loss to GCS
            val_loss = best_artifact.metadata.get(metric_name)
            print("Validation loss: ", val_loss)
            if val_loss is not None:
                val_loss_str = str(val_loss)
                gcs_bucket_name = cfg.cloud.bucket_name_model
                gcs_val_loss_path = os.path.join(cfg.cloud.val_loss_dir, cfg.cloud.val_loss_file)
                upload_string_to_gcs(val_loss_str, gcs_bucket_name, gcs_val_loss_path)
            else:
                print(f"Validation loss '{metric_name}' not found in artifact metadata.")

    except wandb.errors.CommError as e:
        print(f"Error linking artifact: {e}")


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.predict.link_artifacts:
        link_all_artifacts_to_registry(cfg)
    stage_best_model_to_registry(cfg)


if __name__ == "__main__":
    main()
