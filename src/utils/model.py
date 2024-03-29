import torch
import wandb
from torch.nn import DataParallel


def initialize_model(ModelClass, device):
    """
    Creates and returns an instance of the model class.
    Except if there are multiple GPUs available, then the model instance
    is wrapped in an instance of DataParallel.
    """
    model = ModelClass()

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    model.to(device)
    return model


def save_torch_model(model: torch.nn.Module, path: str, model_name: str):
    torch.save(model.state_dict(), path)
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)
