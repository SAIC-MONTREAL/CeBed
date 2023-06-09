"""
Super resolution models
"""
from typing import Dict, Any
import tensorflow as tf
from cebed.models.common import ResidualBlock
from cebed.models.base import BaseModel


class EDSR(BaseModel):
    """
    Implementation of the paper
    Enhanced Deep Residual Networks for single-image super-resolution

    :param hparams : A dictionary of the model's hyperparameters
    """

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        self.conv1 = tf.keras.layers.Conv2D(
            self.hidden_size, self.kernel_size, padding="same"
        )

        self.backbone = tf.keras.Sequential()

        for _ in range(self.n_blocks):
            block = ResidualBlock(
                hidden_size=self.hidden_size, kernel_size=self.kernel_size
            )
            self.backbone.add(block)

        self.backbone.add(
            tf.keras.layers.Conv2D(self.hidden_size, self.kernel_size, padding="same")
        )

        self.upsampling = None

    def build(self, input_shape: tf.TensorShape):
        target_height, target_width, _ = self.output_dim

        if self.upsamling_mode == "deconv":

            pilot_height, pilot_width = input_shape[-3], input_shape[-2]

            height_scale = int(target_height / pilot_height)
            width_scale = int(target_width / pilot_width)

            self.upsampling = tf.keras.layers.Conv2DTranspose(
                self.hidden_size,
                11,
                strides=(height_scale, width_scale),
                padding="same",
            )
        elif self.upsamling_mode == "resize":
            self.upsampling = tf.keras.layers.Resizing(
                target_height, target_width, interpolation="bilinear"
            )

        elif self.upsamling_mode == "identity":
            self.upsampling = tf.keras.layers.Lambda(lambda x: x)
        else:
            raise ValueError("Unsupported upsamling mode")
        self.out = tf.keras.layers.Conv2D(
            self.output_dim[-1], self.kernel_size, padding="same"
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Forward pass

        :param inputs: A batch of inputs
        :param training: True if model is training, False otherwise

        :return upscaled inputs
        """

        inputs = features = self.conv1(inputs)

        features = self.backbone(features)

        features = tf.keras.layers.Add()([inputs, features])

        upscaled = self.upsampling(features)

        return self.out(upscaled)


class ReEsNet(EDSR):
    """
    Implementation of the paper Deep residual learning meets OFDM channel estimation
    :param hparams: A dict of the model's hyperparameters
    """

    def __init__(self, hparams: Dict[str, Any]):
        hparams["upsamling_mode"] = "deconv"
        super().__init__(hparams)


class InReEsNet(EDSR):
    """
    Implementation of the paper
    Low Complexity Channel estimation with Neural Network Solutions
    :param hparams: A dict of the model's hyperparameters
    """

    def __init__(self, hparams: Dict[str, Any]):
        hparams["upsamling_mode"] = "resize"
        super().__init__(hparams)


class MReEsNet(EDSR):
    """
    Implementation of masked ReEsNet
    Instead of low-dimension inputs, the model gets masked inputs

    :param hparams: A dict of the model's hyperparameters
    """

    def __init__(self, hparams: Dict[str, Dict]):
        hparams["upsamling_mode"] = "identity"
        super().__init__(hparams)
