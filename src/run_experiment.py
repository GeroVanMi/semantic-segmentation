import time

import torch
import wandb
from torch.nn import CrossEntropyLoss

from pipeline.evaluate import evaluate_epoch
from pipeline.train import train_epoch
from UNet import UNet
from utils.data import create_train_test_loaders
from utils.model import save_torch_model

NUMBER_OF_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

EXPERIMENT_NAME = "One-hot mask"
PROJECT_NAME = "Lab 03 Semantic Segmentation"
ENTITY_NAME = "computer_vision_01"
ARCHITECTURE_NAME = "U-Net"
DATASET_NAME = "OxfordPets-III"

MODEL_SAVE_PATH = f"../models/{ARCHITECTURE_NAME}_{int(time.time())}.pt"
# In dev mode we only train 3 images for 3 epochs
DEV_MODE = False

if DEV_MODE:
    print("RUNNING IN DEVELOPER TESTING MODE. THIS WILL NOT TRAIN THE MODEL PROPERLY.")
    print("To train the model, set DEV_MODE = False in run_experiment.py!")
    BATCH_SIZE = 3
    NUMBER_OF_EPOCHS = 3
    EXPERIMENT_NAME = f"DEV {EXPERIMENT_NAME}"


def run_experiment():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training on {device} device!")
    input("Confirm with Enter or cancel with Ctrl-C:")

    wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=EXPERIMENT_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": ARCHITECTURE_NAME,
            "batch_size": BATCH_SIZE,
            "dataset": DATASET_NAME,
            "epochs": NUMBER_OF_EPOCHS,
            "dev_mode": DEV_MODE,
            "device": device,
        },
    )
    train_loader, test_loader = create_train_test_loaders(BATCH_SIZE)

    model = UNet().to(device)
    # TODO: Add visualisation of the masks
    # TOOO: Add more interpretable metric like IOU?
    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"#### EPOCH {epoch} ####")
        train_loss = train_epoch(
            model, train_loader, loss_function, optimizer, device, DEV_MODE
        )
        test_loss = evaluate_epoch(model, test_loader, loss_function, device, DEV_MODE)

        wandb.log({"Training Loss": train_loss, "Testing Loss": test_loss})

    save_torch_model(model, MODEL_SAVE_PATH, ARCHITECTURE_NAME)


if __name__ == "__main__":
    run_experiment()
