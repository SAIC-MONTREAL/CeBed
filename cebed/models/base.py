import tensorflow as tf
from typing import Tuple, Dict, Any


class BaseModel(tf.keras.Model):
    """
    The base model object
    :param input_shape: The input shape of the model
    :param output_shape: The model's output shape
    :param hparam: The model hyperparameters
    """

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.hparams = hparams

        for key, val in self.hparams.items():
            setattr(self, key, val)

    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        """
        The logic of one training step given a minibatch of data
        """

        x_batch, y_batch = data

        with tf.GradientTape() as tape:
            preds = self(x_batch)
            loss = self.compiled_loss(y_batch, preds)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(y_batch, preds)

        return {m.name: m.result() for m in self.metrics}
