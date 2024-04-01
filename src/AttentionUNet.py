import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU

from utils.layers import (Attention_Layer, convolution_layer,
                          upsample_convolution)


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = ReLU()

        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = convolution_layer(3, 64)
        self.encoder2 = convolution_layer(64, 128)
        self.encoder3 = convolution_layer(128, 256)
        self.encoder4 = convolution_layer(256, 512)
        self.encoder5 = convolution_layer(512, 1024)

        self.upconv1 = upsample_convolution(1024, 512)
        self.attention1 = Attention_Layer(F_g=512, F_l=512, number_of_features=256)
        # The decoder are always done with two input tensors (because of the skip connections)
        # This is why we have in_channels of 1024 instead 512 here.
        self.decoder1 = convolution_layer(1024, 512)

        self.upconv2 = upsample_convolution(512, 256)
        self.attention2 = Attention_Layer(F_g=256, F_l=256, number_of_features=128)
        self.decoder2 = convolution_layer(512, 256)

        self.upconv3 = upsample_convolution(256, 128)
        self.attention3 = Attention_Layer(F_g=128, F_l=128, number_of_features=64)
        self.decoder3 = convolution_layer(256, 128)

        self.upconv4 = upsample_convolution(128, 64)
        self.attention4 = Attention_Layer(F_g=64, F_l=64, number_of_features=32)
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
        attention1 = self.attention1(upconv1, encoded4)
        skip_connection1 = torch.cat([attention1, upconv1], dim=1)
        decoded1 = self.decoder1(skip_connection1)

        upconv2 = self.upconv2(decoded1)
        attention2 = self.attention2(upconv2, encoded3)
        skip_connection2 = torch.cat([attention2, upconv2], dim=1)
        decoded2 = self.decoder2(skip_connection2)

        upconv3 = self.upconv3(decoded2)
        attention3 = self.attention3(upconv3, encoded2)
        skip_connection3 = torch.cat([attention3, upconv3], dim=1)
        decoded3 = self.decoder3(skip_connection3)

        upconv4 = self.upconv4(decoded3)
        attention4 = self.attention4(upconv4, encoded1)
        skip_connection4 = torch.cat([attention4, upconv4], dim=1)
        decoded4 = self.decoder4(skip_connection4)

        output = self.output_decoder(decoded4)
        return output
