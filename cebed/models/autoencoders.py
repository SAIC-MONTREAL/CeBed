"""
Implementation of Dense Denoising AutoEncoder
"""
from typing import Dict, Any
import tensorflow as tf

from cebed.models.base import BaseModel


class DDAE(BaseModel):
    """
    Implementation of Dense Denoising AutoEncoder similar to [1]
    [1] A Denoising Autoencoder based wireless channel transfer function
    estimator for OFDM communication system
    (https://ieeexplore.ieee.org/document/8669044)

    :param hparams : A dictionary of the model's hyperparameters
    """

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        hidden_sizes = []
        self.backbone = tf.keras.Sequential(name="DDAE")
        # Enconder
        hidden = self.hidden_size

        for _ in range(self.num_layers):
            self.backbone.add(tf.keras.layers.Dense(hidden, activation="relu"))
            hidden /= 2
            hidden_sizes.append(hidden)
        # Decoder

        for hidden in hidden_sizes[::-1]:
            self.backbone.add(tf.keras.layers.Dense(hidden, activation="relu"))

        self.backbone.add(tf.keras.layers.Dense(tf.math.reduce_prod(self.output_dim)))

    def call(self, inputs: tf.Tensor, training=True) -> tf.Tensor:
        """
        Forward pass

        :param inputs: A batch of inputs
        :param training: True if model is training, False otherwise
        :return denoised inputs
        """

        # Flatten inputs
        x = tf.keras.layers.Flatten()(inputs)

        x = self.backbone(x)

        x = tf.keras.layers.Reshape(self.output_dim)(x)

        return x
