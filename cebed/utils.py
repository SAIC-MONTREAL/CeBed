"""
util functions
"""
import os
from typing import Any, TypeVar, Tuple
import random

import numpy as np
import tensorflow as tf
import yaml

T = TypeVar("T")


def set_random_seed(seed: int) -> float:
    """Fix randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def tfinterpolate(
    tensor: tf.Tensor, new_size: Tuple[int], method: str = "bilinear"
) -> tf.Tensor:
    """
    Resize a complex tensor to new size using the specified interpolation method.
    """

    if len(tf.shape(tensor)) < 4:
        tensor = tf.expand_dims(tensor, -1)

    mr = tf.image.resize(tf.math.real(tensor), new_size, method=method)
    mim = tf.image.resize(tf.math.imag(tensor), new_size, method=method)

    return mr.numpy() + 1j * mim.numpy()


def unflatten_last_dim(tensor: tf.Tensor, new_shape: Tuple[int]) -> tf.Tensor:
    """Unflatten the last two dimensions of a tensor into a new size"""
    shape = tf.shape(tensor)
    last_dim = shape[-1]
    tf.debugging.assert_equal(tf.reduce_prod(new_shape), last_dim)
    target_shape = tf.concat([shape[:-1], new_shape], 0)

    return tf.reshape(tensor, target_shape)


def write_metadata(path: str, object_to_save: Any) -> None:
    """Save an object to disk as a yaml file"""
    assert os.path.splitext(path)[1] == ".yaml"
    print(f"Writing metadata to {path}")

    with open(path, "w") as fp:
        yaml.dump(object_to_save, fp)


def read_metadata(path: str) -> T:
    """Load a saved object from a yaml file."""
    with open(path, "r") as fp:
        output = yaml.load(fp, Loader=yaml.Loader)

    return output
