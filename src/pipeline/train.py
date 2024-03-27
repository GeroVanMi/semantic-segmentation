import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    device: str,
):
    model.train()
    total_data_length = len(data_loader)
    epoch_loss = []
    for batch_index, (size, image, mask) in enumerate(data_loader):
        image = torch.Tensor.type(image, dtype=torch.float32)
        image = image.to(device)
        mask_prediction = model(image)

        # Loss calculation with credit to
        # - https://stackoverflow.com/questions/68901153/expected-scalar-type-long-but-found-float-in-pytorch-using-nn-crossentropyloss
        # - https://stackoverflow.com/questions/77475285/pytorch-crossentropy-loss-getting-error-runtimeerror-boolean-value-of-tensor
        mask = mask.squeeze().long()
        loss = loss_function(mask_prediction, mask)
        epoch_loss.append(loss.item())

        # What do these lines really do? This would be interesting to know.
        # Obviously I know that they apply the backpropagation, but what does that mean on a technical level?
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"{batch_index}/{total_data_length} Loss: {loss.item()}")
        break
    return np.mean(epoch_loss)
