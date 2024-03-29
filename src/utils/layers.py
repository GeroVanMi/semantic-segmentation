from torch.nn import BatchNorm2d, Conv2d, ReLU, Sequential


def convolution_layer(input_channels: int, output_channels: int):
    return Sequential(
        Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        BatchNorm2d(output_channels),
        ReLU(),
        Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        BatchNorm2d(output_channels),
        ReLU(),
    )
