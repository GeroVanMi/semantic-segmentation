import torch
import wandb


def save_torch_model(model: torch.nn.Module, path: str, model_name: str):
    torch.save(model.state_dict(), path)
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)
