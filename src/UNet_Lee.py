import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU

from utils.layers import convolution_layer, upsample_convolution


class UNet_Lee(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

        # Encoder Path
        self.pool = MaxPool2d(2, stride=2)

        self.encoder1 = convolution_layer(3, 64)
        self.encoder2 = convolution_layer(64, 128)
        self.encoder3 = convolution_layer(128, 256)
        self.encoder4 = convolution_layer(256, 512)
        self.encoder5 = convolution_layer(512, 1024)

        # Decoder Path
        # The decoder are always done with two input tensors (because of the skip connections)
        # This is why we have in_channels of 1024 instead 512 here.
        self.upconv1 = upsample_convolution(1024, 512)
        self.decoder1 = convolution_layer(1024, 512)

        self.upconv2 = upsample_convolution(512, 256)
        self.decoder2 = convolution_layer(512, 256)

        self.upconv3 = upsample_convolution(256, 128)
        self.decoder3 = convolution_layer(256, 128)

        self.upconv4 = upsample_convolution(128, 64)
        self.decoder4 = convolution_layer(128, 64)

        self.output_decoder = Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        encoded1 = self.encoder1(x)
        pool1 = self.pool(encoded1)

        encoded2 = self.encoder2(pool1)
        pool2 = self.pool(encoded2)

        encoded3 = self.encoder3(pool2)
        pool3 = self.pool(encoded3)

        encoded4 = self.encoder4(pool3)
        pool4 = self.pool(encoded4)

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
