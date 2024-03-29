from torch.nn import (BatchNorm2d, Conv2d, Module, ReLU, Sequential, Sigmoid,
                      Upsample)


def convolution_layer(input_channels: int, output_channels: int):
    return Sequential(
        Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        BatchNorm2d(output_channels),
        ReLU(),
        Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        BatchNorm2d(output_channels),
        ReLU(),
    )


def upsample_convolution(input_channels: int, output_channels: int) -> Sequential:
    return Sequential(
        Upsample(scale_factor=2),
        Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        ),
        BatchNorm2d(output_channels),
        ReLU(),
    )


class Attention_Layer(Module):
    """
    Implementation by Lee Jun Hyun, adapted by Gérôme Meyer.
    https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    """

    def __init__(self, F_g, F_l, number_of_features):
        """
        TODO: What does F_g stand for?
        TODO: What does F_l stand for?
        """
        super(Attention_Layer, self).__init__()
        self.W_g = Sequential(
            Conv2d(
                F_g, number_of_features, kernel_size=1, stride=1, padding=0, bias=True
            ),
            BatchNorm2d(number_of_features),
        )

        self.W_x = Sequential(
            Conv2d(
                F_l, number_of_features, kernel_size=1, stride=1, padding=0, bias=True
            ),
            BatchNorm2d(number_of_features),
        )

        self.psi = Sequential(
            Conv2d(
                number_of_features, 1, kernel_size=1, stride=1, padding=0, bias=True
            ),
            BatchNorm2d(1),
            Sigmoid(),
        )

        self.relu = ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
