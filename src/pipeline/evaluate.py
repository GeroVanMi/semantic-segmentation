import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


def evaluate_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    device: str,
    dev_mode=False,
):
    # acc_met = Accuracy(task="multiclass", num_classes=3).to(DEVICE)
    # lss_met = MeanMetric().to(DEVICE)
    model.eval()
    epoch_loss = []
    total_data_length = len(data_loader)
    with torch.inference_mode():
        for batch_index, (size, image, mask) in enumerate(data_loader):
            image = torch.Tensor.type(image, dtype=torch.float32)
            image = image.to(device)
            mask_prediction = model(image)

            # Loss calculation with credit to
            # - https://stackoverflow.com/questions/68901153/expected-scalar-type-long-but-found-float-in-pytorch-using-nn-crossentropyloss
            # - https://stackoverflow.com/questions/77475285/pytorch-crossentropy-loss-getting-error-runtimeerror-boolean-value-of-tensor
            # mask = mask.squeeze(dim=1).long()
            mask = mask.to(device)

            loss = loss_function(mask_prediction, mask)
            epoch_loss.append(loss.item())

            # acc_met.update(pred, y)
            # lss_met.update(loss_fn(pred, y))
            if batch_index + 1 % 100 == 0:
                print(f"{batch_index + 1}/{total_data_length} Test Loss: {loss.item()}")

            if dev_mode:
                print(f"Test Loss: {loss.item()}")
                break

    return np.mean(epoch_loss)
