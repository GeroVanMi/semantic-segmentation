import torch
import wandb
from torch.nn import CrossEntropyLoss

from PetsDataset import create_dataloader
from pipeline.train import train_epoch
from UNet import UNet

NUMBER_OF_EPOCHS = 1
BATCH_SIZE = 3
LEARNING_RATE = 0.001


def run_experiment():
    wandb.init(
        # set the wandb project where this run will be logged
        project="Lab 03 Semantic Segmentation",
        entity="computer_vision_01",
        name="UNet testing experiment",
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "U-Net",
            "dataset": "OxfordPets-III",
            "epochs": NUMBER_OF_EPOCHS,
        },
    )
    oxford_pets_loader = create_dataloader(BATCH_SIZE)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = UNet().to(device)
    # TODO: Add W&B logging
    # TODO: Add Train / Test Split
    # TODO: Add visualisation of the masks
    # TOOO: Add more interpretable metric like IOU?
    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"#### EPOCH {epoch} ####")
        epoch_loss = train_epoch(
            model, oxford_pets_loader, loss_function, optimizer, device
        )
        wandb.log({"loss": epoch_loss})


if __name__ == "__main__":
    run_experiment()
