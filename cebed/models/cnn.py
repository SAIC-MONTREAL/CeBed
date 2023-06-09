import tensorflow as tf
from typing import Tuple

from cebed.models.base import BaseModel
from cebed.models.common import ConvBlock


def create_srcnn_model(
    num_channels: int,
    hidden_size: Tuple[int] = (64, 32),
    kernel_size: Tuple[int] = (9, 1, 5),
) -> tf.keras.Sequential:
    """
    Implementation of the SRCNN model
    [Paper] Image Super-Resolution Using Deep Convolutional Networks
    [URL] https://arxiv.org/abs/1501.00092
    :param num_channels: The number of output channels
    :param hidden_size: A list of the number units of the hidden layers
    :param kernel_size: The size of the kernel used in the conv layer
    :return A SRCNN model
    """

    srcnn = tf.keras.Sequential(name="SRCNN")

    for i, size in enumerate(hidden_size):
        srcnn.add(
            tf.keras.layers.Conv2D(
                size,
                kernel_size[i],
                activation="relu",
                kernel_initializer="he_normal",
                padding="same",
            )
        )

    srcnn.add(
        tf.keras.layers.Conv2D(
            num_channels,
            kernel_size[-1],
            kernel_initializer="he_normal",
            padding="same",
        )
    )

    return srcnn


def create_dncnn_model(
    num_channels: int, num_layers: int = 18, hidden_size: int = 64, kernel_size: int = 3
) -> tf.keras.Sequential:
    """
    Implementation of the DnCNN model
    [Paper] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
    [URL] https://ieeexplore.ieee.org/document/7839189
    :param num_channels: The number of output channels
    :param num_layers: The number of hidden layers
    :param hidden_size: The number of units in each hidden layer
    :param kernel_size: The size of the kernel used in the conv layer
    :return A DnCnn model
    """
    dccnn = tf.keras.Sequential(name="DcCNN")
    dccnn.add(
        tf.keras.layers.Conv2D(
            hidden_size, kernel_size, activation="relu", padding="same"
        )
    )

    for _ in range(num_layers):
        dccnn.add(ConvBlock(hidden_size))

    dccnn.add(tf.keras.layers.Conv2D(num_channels, kernel_size, padding="same"))

    return dccnn


class ChannelNet(BaseModel):
    """
    Implementation of Deep Learning-Based Channel Estimation
    [URL] https://arxiv.org/abs/1810.05893
    :param hparams : A dictionary of the model's hyperparameters
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        if self.int_type not in ["bilinear", "bicubic", "nearest"]:
            raise ValueError(f"Interpolation type not supported {self.int_type}")

        # The super-resolution model
        self.sr_model = create_srcnn_model(
            num_channels=self.output_dim[-1],
            hidden_size=self.sr_hidden_size,
            kernel_size=self.sr_kernels,
        )
        # The denoising model
        self.denoiser = create_dncnn_model(
            num_channels=self.output_dim[-1],
            hidden_size=self.dc_hidden,
            num_layers=self.num_dc_layers,
        )

    def call(self, inputs):
        x = tf.keras.layers.Resizing(
            self.output_dim[0], self.output_dim[1], interpolation=self.int_type
        )(inputs)

        x = self.sr_model(x)

        noise = self.denoiser(x)

        return x - noise
