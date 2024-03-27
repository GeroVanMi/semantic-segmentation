import torch.nn as nn
from torch.nn import Conv2d, ReLU


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = ReLU()
        self.encoder1 = Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)

        self.decoder1 = Conv2d(20, 3, 3, padding=1)
        # TODO: Add skip connections with torch.cat()

    def forward(self, x):
        encoded_x1 = self.relu(self.encoder1(x))
        decoded_x1 = self.decoder1(encoded_x1)
        return decoded_x1
