import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, ReLU, Sequential


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = ReLU()

        self.encoder1 = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(64, 64, 3, padding=1),
            ReLU(),
        )
        self.pool1 = MaxPool2d(2, stride=2)

        self.encoder2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(128, 128, 3, padding=1),
            ReLU(),
        )
        self.pool2 = MaxPool2d(2, stride=2)

        self.encoder3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, 3, padding=1),
            ReLU(),
        )
        self.pool3 = MaxPool2d(2, stride=2)

        self.encoder4 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(512, 512, 3, padding=1),
            ReLU(),
        )
        self.pool4 = MaxPool2d(2, stride=2)

        self.encoder5 = Sequential(
            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
        )

        self.upconv1 = ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # The decoder are always done with two input tensors (because of the skip connections)
        # This is why we have in_channels of 1024 instead 512 here.
        self.decoder1 = Sequential(
            Conv2d(1024, 512, 3, padding=1),
            ReLU(),
            Conv2d(512, 512, 3, padding=1),
            ReLU(),
        )

        self.upconv2 = ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = Sequential(
            Conv2d(512, 256, 3, padding=1),
            ReLU(),
            Conv2d(256, 256, 3, padding=1),
            ReLU(),
        )

        self.upconv3 = ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = Sequential(
            Conv2d(256, 128, 3, padding=1),
            ReLU(),
            Conv2d(128, 128, 3, padding=1),
            ReLU(),
        )

        self.upconv4 = ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = Sequential(
            Conv2d(128, 64, 3, padding=1),
            ReLU(),
            Conv2d(64, 64, 3, padding=1),
            ReLU(),
        )

        self.output_decoder = Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        # TODO: Add skip connections with torch.cat()

        encoded1 = self.encoder1(x)
        pool1 = self.pool1(encoded1)

        encoded2 = self.encoder2(pool1)
        pool2 = self.pool2(encoded2)

        encoded3 = self.encoder3(pool2)
        pool3 = self.pool3(encoded3)

        encoded4 = self.encoder4(pool3)
        pool4 = self.pool4(encoded4)

        encoded5 = self.encoder5(pool4)

        upconv1 = self.upconv1(encoded5)
        skip_connection1 = torch.cat([upconv1, encoded4], dim=1)
        decoded1 = self.decoder1(skip_connection1)

        upconv2 = self.upconv2(decoded1)
        skip_connection2 = torch.cat([upconv2, encoded3], dim=1)
        decoded2 = self.decoder2(skip_connection2)

        upconv3 = self.upconv3(decoded2)
        skip_connection3 = torch.cat([upconv3, encoded2], dim=1)
        decoded3 = self.decoder3(skip_connection3)

        upconv4 = self.upconv4(decoded3)
        skip_connection4 = torch.cat([upconv4, encoded1], dim=1)
        decoded4 = self.decoder4(skip_connection4)

        output = self.output_decoder(decoded4)
        return output
