import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple


from cebed.models.common import ResidualBlock, GlobalSelfAttention, FeedForward
from cebed.models.base import BaseModel


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder layer as decribed in https://arxiv.org/abs/1706.03762

    :param key_dim: The size of each attention head for query and key
    :param num_heads: The number of attention heads
    :param ff_dim: The hidden dimension of the FF module
    :param dropout_rate: The dropout rate

    """

    def __init__(
        self,
        key_dim: int = 16,
        num_heads: int = 2,
        ff_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
        )

        self.ffn = FeedForward(key_dim=ff_dim, ff_dim=ff_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # [B, N, key_dim*num_head]
        x = self.self_attention(x)
        # [B, N, ff_dim]
        x = self.ffn(x)

        return x


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder block
    :param num_layers: Number of encoder layers
    :param key_dim: The size of each attention head for query and key
    :param num_heads: The number of attention heads
    :param ff_dim: The hidden dimension of the FF module
    :param dropout_rate: The dropout rate
    """

    def __init__(
        self,
        num_layers: int = 2,
        key_dim: int = 16,
        num_heads: int = 2,
        ff_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.enc_layers = [
            EncoderLayer(
                key_dim=key_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The layer logic
        """

        for layer in self.enc_layers:
            x = layer(x)

        return x


class HA02Decoder(tf.keras.layers.Layer):
    """
    The HA02 Decoder: Residual blocks + Upsampling block
    :param num_layers: Number of decoder layers
    :param hidden_size: The hidden size for the residual block
    :param kernel_size: The convolution kernel size
    """

    def __init__(self, num_layers: int = 2, hidden_size: int = 2, kernel_size: int = 2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(hidden_size, kernel_size, padding="same")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.blocks = [
            ResidualBlock(
                hidden_size=self.hidden_size,
                kernel_size=self.kernel_size,
                layernorm=True,
            )
            for _ in range(num_layers)
        ]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)

        for block in self.blocks:
            x = block(x)
        return x


class UpsamlingBlock(tf.keras.layers.Layer):
    """
    Upsampling block for HA02 architecture

    :param output_dim: A tuple of the output shape
    :param kernel_size: The convolution kernel size
    """

    def __init__(self, output_dim: Tuple[int], kernel_size: int = 2):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                # [batch, 2, hidden_size, nps*npf]
                tf.keras.layers.Permute((2, 3, 1)),
                # [batch, 2, hidden_size, ns*nf]
                tf.keras.layers.Dense(output_dim[0] * output_dim[1]),
                # [batch, ns*nf, 2, hidden_size]
                tf.keras.layers.Permute((3, 1, 2)),
                # [batch, ns*nf, 2, 1]
                tf.keras.layers.Conv2D(1, kernel_size, padding="same"),
            ]
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Layer logic
        """

        return self.seq(inputs)


class HA02(BaseModel):
    """
    Implementation of the paper Attention Based Neural Networks for Wireless
    Channel Estimation
    https://arxiv.org/pdf/2204.13465.pdf
    """

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        self.encoder = None
        self.decoder = None
        self.upsamling = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build model's layers
        """
        # Number of heads in the number of pilot symbols
        num_heads = input_shape[1]
        # head size is set to be equal the number of pilot subcarriers
        head_size = input_shape[2]

        # ff_dim = num_pilot_symbols*num_pilot_subcarriers
        # We follow the original transformer paper
        # and keep the dimensions of all the sub-layers eqaul
        ff_dim = np.prod(input_shape[1:-1])

        self.encoder = Encoder(
            num_layers=self.num_en_layers,
            key_dim=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=self.dropout_rate,
        )

        self.decoder = HA02Decoder(
            self.num_dc_layers, self.hidden_size, kernel_size=self.kernel_size
        )

        self.upsamling = UpsamlingBlock(self.output_dim, self.kernel_size)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """
        Forward pass

        :param inputs: A batch of inputs
        :param training: True if model is training, False otherwise

        :return upscaled inputs
        """
        # [batch, nps, nfs, c]
        shape = inputs.shape
        # [batch, nps*npf, c]
        inputs = tf.keras.layers.Reshape((-1, shape[-1]))(inputs)

        # NOTE in the original paper,
        # the embedding size = n pilot symbols*n pilots subcarriers
        # We are using TF MultiHeadAttention where the emb dimension is
        # the channel dim, that is why we are permuting the inputs
        # [batch, c, nps*npf]
        inputs = tf.keras.layers.Permute((2, 1))(inputs)

        # [batch, c, nps*npf]
        latent = self.encoder(inputs)

        # [batch, nps*npf, c]
        latent = tf.keras.layers.Permute((2, 1))(latent)

        shape = latent.shape
        # Reshape before sending to the decoder
        # [batch, nps*npf, c, 1]
        latent = tf.keras.layers.Reshape([shape[1], shape[2], 1])(latent)

        # [batch, nps*npf, c, hidden_size]
        decoded = self.decoder(latent)

        upscaled = self.upsamling(decoded)

        outputs = tf.keras.layers.Reshape(self.output_dim)(upscaled)

        return outputs


class MTRE(BaseModel):
    """
    Implementation of the paper Channel Estimation Method Based on Transformer
    in High Dynamic Environment
    https://ieeexplore.ieee.org/abstract/document/9299821
    """

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Conv2D(
            input_shape[-1], kernel_size=1, padding="same"
        )

        num_heads = input_shape[-1]
        head_size = input_shape[-2]
        ff_dim = num_heads * head_size
        self.encoder = tf.keras.Sequential(name="Encoder")

        for _ in range(self.num_layers):
            self.encoder.add(
                EncoderLayer(
                    head_size, num_heads, ff_dim, dropout_rate=self.dropout_rate
                )
            )
        self.out = tf.keras.layers.Reshape(self.output_dim)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # NOTE we dont use activation in the embedding layer since it gives better results

        embs = self.embedding(inputs)

        # NOTE in the original paper the attention is done on frequency only
        # [batch, ns, nf*c]
        embs = tf.keras.layers.Reshape((embs.shape[1], -1))(embs)

        outputs = self.encoder(embs)

        outputs = self.out(outputs)

        return outputs
