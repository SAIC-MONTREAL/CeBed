"""Test utils functions"""
import pytest
import tensorflow as tf
from cebed import utils


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_tfinterpolate():
    shape = (5, 2, 72, 8)
    real = tf.random.uniform(shape=shape, dtype=tf.float32)
    img = tf.random.uniform(shape=shape, dtype=tf.float32)

    tensor = tf.complex(real, img)

    resized = utils.tfinterpolate(tensor, (14, 72))

    assert resized.dtype == tf.complex64
    assert resized.shape == (5, 14, 72, 8)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_unflatten_last_dim():
    shape = (1, 1, 72)
    real = tf.random.uniform(shape=shape, dtype=tf.float32)

    resized = utils.unflatten_last_dim(real, (2, 36))

    assert resized.shape == (1, 1, 2, 36)
    assert resized.dtype == real.dtype
