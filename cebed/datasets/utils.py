"""Helper functions for dataset manipulation"""

from typing import List, Dict, Callable
import h5py
import pickle

import numpy as np
from scipy.io import loadmat
import tensorflow as tf

###############################################################################
# Data I/O
###############################################################################


def read_dataset_from_file(filename, keys=None):
    """Helper function to read dataset from disk"""

    if filename.endswith(".pkl"):
        return read_dataset_from_pickle(filename)

    if filename.endswith(".mat"):
        return loadmat(filename)

    if filename.endswith(".hdf5"):
        return h5py.File(filename)

    raise NotImplementedError("Unsupported file type")


def read_dataset_from_pickle(
    filename: str, keys: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Read dataset from pickle file
    :param filename: The dataset filename
    """
    if keys is None:
        keys = ["x", "y", "h"]

    with open(filename, "rb") as f:
        data = pickle.load(f)

    outputs = {}

    for key in keys:
        sources = [v.numpy() for k, v in data.items() if k.startswith(f"{key}_")]
        outputs[key] = np.array(sources)
    return outputs


###############################################################################
# Data preprocessing/manipulation
###############################################################################
def make_split(data_size, train_fraction=0.9):
    test_fraction = 1 - train_fraction
    split = int(data_size * test_fraction)
    indices = list(range(data_size))
    np.random.shuffle(indices)
    train_indices = indices[split + 1 :]
    test_indices = indices[: split + 1]

    return train_indices, test_indices


def combine_datasets(sources):
    """
    Combines a list of data sources along the first axis
    """

    return np.concatenate(sources, axis=0)


def convert_complex(inputs: tf.Tensor) -> tf.Tensor:
    """
    Convert the complex input to real
    by stacking the real and imaginary parts along the last dimension
    """

    real = tf.math.real(inputs)
    imaginary = tf.math.imag(inputs)

    if len(inputs.shape) < 3:
        real = tf.expand_dims(real, axis=-1)
        imaginary = tf.expand_dims(imaginary, axis=-1)

    inputs = tf.concat([real, imaginary], -1)

    return inputs


def complex_to_real(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert a complex tensor to a real tensor
    Stack the real and imaginary parts along the last dimension

    :param tensor: a 3D or 2D complex tensor [h,w,c] or [h,w]

    :return: real tensor [h,w,c*2]
    """

    real = tf.math.real(tensor)
    imaginary = tf.math.imag(tensor)

    if len(tensor.shape) < 3:
        real = tf.expand_dims(real, axis=-1)
        imaginary = tf.expand_dims(imaginary, axis=-1)

    real_tensor = tf.concat([real, imaginary], -1)

    return real_tensor


def real_to_complex(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert a real tensor to a complex tensor
    Assume that the real and imaginary parts are stacked  along the last dimension

    :param tensor: a 3D real tensor [h,w,c]

    :return: complex tensor [h,w,c/2]
    """

    if tensor.shape[-1] == 2:
        c_mat = tf.complex(tensor[:, :, 0], tensor[:, :, 1])

        return c_mat
    elif tensor.shape[-1] > 2:
        nc = int(tensor.shape[-1] / 2)
        c_mat = tf.complex(tensor[:, :, :nc], tensor[:, :, nc:])

        return c_mat
    else:
        raise ValueError()


def prepare_dataset(
    ds: tf.data.Dataset,
    batch_size: int = 256,
    shuffle: bool = False,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] = None,
    buffer_size: int = 1024,
    drop_remainder: bool = False,
) -> tf.data.Dataset:

    """Prepare a dataset for training and eval"""

    if preprocess_fn is not None:
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def postprocess(inputs: tf.Tensor) -> tf.Tensor:
    # From real inputs to complex
    inputs = real_to_complex(inputs)

    # Transpose

    if len(inputs.shape) == 3:
        inputs = tf.transpose(inputs, (2, 0, 1))

    return inputs
