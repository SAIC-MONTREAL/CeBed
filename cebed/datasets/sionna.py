"""
Defines dataset classes
"""
import os

from typing import Tuple, Optional
from functools import partial
import numpy as np
import tensorflow as tf
from copy import deepcopy
import glob
from cebed.datasets.utils import (
    complex_to_real,
    make_split,
    prepare_dataset,
    read_dataset_from_file,
    combine_datasets,
)
from cebed.datasets.base import TensorOfflineDataset
from cebed.envs import OfdmEnv, EnvConfig
from cebed.utils import read_metadata


def preprocess_inputs(
    inputs: tf.Tensor, input_type="low", mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Model inputs preprocessing.
    Expects an OFDM frame and **not** a batch of frames
    1- Reshaping operations:
        - Remove unecessary dimensions in the input
        - if num rx antennas >1 or num_ues >1,
          these dimensions are added to the channel dimension
    2- Depending on the model input type, prepare the inputs
        "raw": returns the frame as it is.
        **Note**, the frame is masked at non pilot positions
        "low": applies a mask to the frame and reduces its dimensions
    3- Convert from complex to real values

    :param inputs: a complex tensor
    [num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]

    :param input_type: a string to specific to type of the input
    :param mask: a tensor where only the pilot locations are equal to 1
    shape = [n_t, n_t_streams, n_symbols, n_subcarriers]

    :return: a real tensor
    """

    if input_type == "low" and mask is None:
        raise ValueError(f"Mask is needed for the requested input type {input_type}")

    if len(inputs.shape) > 6:
        raise ValueError("Input shape can to have more than 6 dimensions")

    # Remove dimensions that are equal to 1
    x = tf.squeeze(inputs)

    # Stack the nr and nt dimensions if they are different than 1

    if len(x.shape) == 4:
        # Stack the nr and nt dimensions
        nr, nt, ns, nf = x.shape
        x = tf.reshape(x, (nr * nt, ns, nf))

    # [num_symbols, num_subcarriers, num_channels]

    if len(x.shape) > 2:
        x = tf.transpose(x, [1, 2, 0])

    if input_type == "raw":
        pre_x = x
    elif input_type == "low":
        pilot_indices = tf.where(mask)
        # gather the pilot symbols
        symbol_indices, _ = tf.unique(pilot_indices[:, -2])
        low_x = tf.gather(indices=symbol_indices, params=x, axis=0)
        # gather the pilot subcarriers
        subc_indices, _ = tf.unique(pilot_indices[:, -1])
        pre_x = tf.gather(indices=subc_indices, params=low_x, axis=1)
    else:
        raise ValueError(f"Unknown input mode {input_type}")

    # Convert to real
    pre_x = complex_to_real(pre_x)

    return pre_x


def preprocess_labels(labels: tf.Tensor):
    """
    Per sample preprocessing function for tf dataset
    This works on samples and **not** on a batch

    :param labels: 6-D complex tensor
    [num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]

    :return: A real tensor
    """

    # [num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]

    if len(labels.shape) > 6:
        raise ValueError("Input shape can to have more than 6 dimensions")

    labels = tf.squeeze(labels)

    # [num_channels, num_symbols, num_subcarriers]

    if len(labels.shape) == 4:
        # Stack the nr and nt dimensions
        nr, nt, ns, nf = labels.shape
        labels = tf.reshape(labels, (-1, ns, nf))

    # [num_symbols, num_subcarriers, num_channels]

    if len(labels.shape) > 2:
        labels = tf.transpose(labels, [1, 2, 0])

    # [num_symbols, num_subcarriers, num_channels*2]

    return complex_to_real(labels)


class OfflineSionnaDataset(TensorOfflineDataset):
    """Offline single domain Sionna dataset"""

    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        input_type: str = "low",
        seed: int = 0,
    ):

        super().__init__(
            path=path, seed=seed, train_split=train_split, input_type=input_type
        )

    def setup(self):
        """Initial Setup"""
        # Create ofdm environment
        self.create_env()

        self.mask = self.env.get_mask()

        # self.load_data()

        # self.split()
        super().setup()

    def create_env(self):
        """Create OFDM env using saved metadata"""
        assert os.path.isfile(os.path.join(self.main_path, "metadata.yaml"))
        env_config = read_metadata(os.path.join(self.main_path, "metadata.yaml"))

        if isinstance(env_config, dict):
            saved_config = deepcopy(env_config)
            env_config = EnvConfig()
            env_config.from_dict(saved_config)
        self.env = OfdmEnv(env_config)

    def load_data(self):
        """
        Reads dataset from disk.
        """

        data_path = glob.glob(f"{self.main_path}/data*")[0]

        data = read_dataset_from_file(data_path)

        self.labels = np.array(data["h"][:])
        self.x_samples = np.array(data["x"][:])
        self.y_samples = np.array(data["y"][:])
        self.inputs = self.env.estimate_at_pilot_locations(self.y_samples).numpy()

        self.size = self.labels.shape[0]

    def preprocess_inputs(self, inputs):
        return preprocess_inputs(inputs, input_type=self.input_type, mask=self.mask)

    def preprocess(self, inputs, labels, train=True):
        """
        Per sample preprocessing function passed to tf.data.Dataset.map
        :param inputs: 6-D complex tensor
        [num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        :param labels: 6-D complex tensor
        [num_r, num_r_ants, n_t, n_t_streams, num_symbols, num_subcarriers]
        :return: Preprocessed inputs and labels
        """

        if train:
            labels = preprocess_labels(labels)

        inputs = self.preprocess_inputs(inputs)

        return inputs, labels

    @property
    def output_shape(self):
        return (
            self.env.config.num_ofdm_symbols,
            self.env.rg.num_effective_subcarriers,
            self.env.config.num_rx_antennas * self.env.config.n_ues * 2,
        )

    def get_input_shape(self):
        """Return the input data shape"""

        if self.input_type == "raw":
            return (
                self.env.config.num_ofdm_symbols,
                self.env.rg.num_effective_subcarriers,
                self.env.config.num_rx_antennas * self.env.config.n_ues * 2,
            )

        if self.input_type == "low":
            return (
                self.env.n_pilot_symbols,
                self.env.n_pilot_subcarriers,
                self.env.config.num_rx_antennas * self.env.config.n_ues * 2,
            )
        raise ValueError(f"Unknown input type {self.input_type}")

    @property
    def pilots(self):
        return self.env.rg.pilot_pattern.pilots

    @property
    def num_symbols(self):
        return self.env.config.num_ofdm_symbols

    @property
    def num_subcarries(self):
        return self.env.config.fft_size

    @property
    def num_pilot_symbols(self):
        return self.env.n_pilot_symbols

    @property
    def num_pilot_subcarriers(self):
        return self.env.n_pilot_subcarriers


class MultiDomainOfflineSionnaDataset(OfflineSionnaDataset):
    """
    Multi-domain Sionna dataset
    The dataset contains N domains and each domain consists of X number of samples
    All the domains are used for training
    Example: N=5, X=15K
    the channel data will have the shape [N, X, n_bs, nr, nt, nt_streams, ns, nf]
    :param path: the dataset path. the folder should contain the data and metadata files
    :param train_split: the fraction of data allocated to training
    :param input_type: the input type. Available choices ["raw", "low"]
    """

    def __init__(
        self,
        path: str,
        train_split: float = 0.9,
        input_type: str = "low",
        seed: int = 0,
    ):

        self.num_domains = None
        super().__init__(
            path=path, seed=seed, train_split=train_split, input_type=input_type
        )
        self.seed = seed
        self.input_type = input_type
        self.train_split = train_split
        self.main_path = path

        self.setup()

    def load_data(self):
        """
        Reads dataset from disk.
        """
        data_path = glob.glob(f"{self.main_path}/data*")[0]

        data = read_dataset_from_file(data_path)

        self.labels = np.array(data["h"][:])
        self.x_samples = np.array(data["x"][:])
        self.y_samples = np.array(data["y"][:])

        self.inputs = []
        self.num_domains = self.labels.shape[0]
        self.size = np.prod(self.labels.shape[:2])

        for d in range(self.num_domains):
            ds_inputs = self.env.estimate_at_pilot_locations(self.y_samples[d]).numpy()
            self.inputs.append(ds_inputs)

    def split(self):
        """Splits the dataset for each domain into train, val and test sets"""

        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []

        self.test_indices = []

        for ds in range(self.num_domains):
            # split the data into train and test

            train_indices, test_indices = make_split(
                len(self.inputs[ds]), train_fraction=self.train_split
            )

            assert len(train_indices) > 1, "train split cannot be empty"

            # split train data into train and validation sets
            train_indices, val_indices = make_split(
                len(train_indices), self.train_split
            )

            # For train and validation, the inputs are the LS estimates
            train_x.append(self.inputs[ds][train_indices])
            train_y.append(self.labels[ds][train_indices])
            val_x.append(self.inputs[ds][val_indices])
            val_y.append(self.labels[ds][val_indices])

            # For testing, we need the transmitted and received signals
            # for baseline evaluation
            test_x.append(self.inputs[ds][test_indices])
            test_y.append(self.labels[ds][test_indices])

            self.test_indices.append(test_indices)

        # Train using multiple domains (e.g., SNRs)
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.val_x = np.array(val_x)
        self.val_y = np.array(val_y)
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)

    def get_loaders(
        self, train_batch_size: int, eval_batch_size: int
    ) -> Tuple[tf.data.Dataset]:
        """Get the data loaders for train and validation"""
        train_loader = self.get_train_loader(train_batch_size)
        eval_loader = self.get_eval_loader(eval_batch_size, setname="val")

        return train_loader, eval_loader

    def get_train_loader(self, batch_size: int) -> tf.data.Dataset:
        """
        Prepare train dataset. Each batch is sampled uniformly across different
        train domains
        """
        train_datasets = []

        for i in range(self.num_domains):
            ds = tf.data.Dataset.from_tensor_slices((self.train_x[i], self.train_y[i]))
            train_datasets.append(ds)
        train_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
        train_ds = prepare_dataset(
            train_ds, batch_size=batch_size, shuffle=True, preprocess_fn=self.preprocess
        )

        return train_ds

    def get_eval_loader(
        self, batch_size: int = 32, setname: str = "val"
    ) -> tf.data.Dataset:
        """
        Prepare val dataset.
        The data from all domains are concatenated and will be passed sequentially
        to the model
        """

        if setname == "val":
            inputs = self.val_x
            outputs = self.val_y
        else:
            inputs = self.test_x
            outputs = self.test_y

        inputs = combine_datasets(inputs)
        outputs = combine_datasets(outputs)

        preprocess_fn = partial(self.preprocess, train=setname == "val")

        eval_ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        eval_ds = prepare_dataset(
            eval_ds, batch_size=batch_size, shuffle=False, preprocess_fn=preprocess_fn
        )

        return eval_ds
