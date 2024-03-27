import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from PetsDataset import create_dataloader
from UNet import UNet

NUMBER_OF_EPOCHS = 2
LEARNING_RATE = 0.001


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    optimizer: torch.optim.SGD,
):
    model.train()
    total_data_length = len(data_loader)
    for batch_index, (size, image, mask) in enumerate(data_loader):
        image = torch.Tensor.type(image, dtype=torch.float32)
        mask_prediction = model(image)

        # Loss calculation with credit to
        # - https://stackoverflow.com/questions/68901153/expected-scalar-type-long-but-found-float-in-pytorch-using-nn-crossentropyloss
        # - https://stackoverflow.com/questions/77475285/pytorch-crossentropy-loss-getting-error-runtimeerror-boolean-value-of-tensor
        mask = mask.squeeze().long()
        loss = loss_function(mask_prediction, mask)

        # What do these lines really do? This would be interesting to know.
        # Obviously I know that they apply the backpropagation, but what does that mean on a technical level?
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"{batch_index}/{total_data_length} Loss: {loss.item()}")


def main():
    data_loader = create_dataloader(64)

    model = UNet()
    # TODO: Add W&B logging
    # TODO: Add Train / Test Split
    # TODO: Add visualisation of the masks
    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"#### EPOCH {epoch} ####")
        train_epoch(model, data_loader, loss_function, optimizer)


if __name__ == "__main__":
    main()
