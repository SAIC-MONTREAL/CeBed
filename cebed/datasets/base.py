"""
Base dataset classes
"""

from typing import Tuple

import tensorflow as tf

from cebed.datasets.utils import (
    make_split,
    complex_to_real,
    prepare_dataset,
)


class TensorOfflineDataset:
    """Abstract class for single domain offline dataset"""

    def __init__(
        self, path: str, seed=int, train_split: float = 0.9, input_type: str = "raw"
    ) -> None:

        self.main_path = path
        self.seed = seed
        self.train_split = train_split
        self.input_type = input_type

        self.env = None
        # NN labels and inputs
        self.labels = None
        self.inputs = None
        # Transmit and received symbols
        self.x_samples = None
        self.y_samples = None

        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None

        self.size = None

        self.setup()

    def get_mask(self) -> tf.Tensor:
        """Define the pilot mask"""
        raise NotImplementedError

    def setup(self):
        """Initial setup"""
        self.load_data()
        self.split()
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))

    def val_ds(self, setname="val"):
        """Define the val dataset"""

        if setname == "val":
            inputs = self.val_x
            outputs = self.val_y
        else:
            inputs = self.test_x
            outputs = self.test_y

        return tf.data.Dataset.from_tensor_slices((inputs, outputs))

    def __len__(self) -> int:
        """Returns the size of the whole dataset"""

        if self.size is not None:
            return self.size

        return 0

    def load_data(self, *args, **kwargs) -> None:
        """Read dataset from disk"""
        raise NotImplementedError

    def preprocess(self, input: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor]:
        """
        Per sample preprocessing function passed to tf.data.Dataset.map
        :param input: a complex tensor
        :param label: acomplex tensor
        :return: Preprocessed input and label
        """

        label = complex_to_real(label)
        input = complex_to_real(input)

        return input, label

    def split(self):
        """Split into train, val, test datasets"""
        train_indices, self.test_indices = make_split(
            len(self.inputs), train_fraction=self.train_split
        )

        # split train data into train and validation sets
        train_indices, val_indices = make_split(len(train_indices), self.train_split)
        # For train and validation, the inputs are the LS estimates
        self.train_x = self.inputs[train_indices]
        self.train_y = self.labels[train_indices]
        self.val_x = self.inputs[val_indices]
        self.val_y = self.labels[val_indices]
        self.test_x = self.inputs[self.test_indices]
        self.test_y = self.labels[self.test_indices]

    def get_loaders(
        self, train_batch_size: int, eval_batch_size: int
    ) -> Tuple[tf.data.Dataset]:
        """Make the training loaders"""

        train_loader = prepare_dataset(
            self.train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            preprocess_fn=self.preprocess,
        )
        eval_loader = self.get_eval_loader(eval_batch_size)

        return train_loader, eval_loader

    def get_eval_loader(self, batch_size: int, setname: str = "val") -> tf.data.Dataset:

        return prepare_dataset(
            self.val_ds(setname),
            batch_size=batch_size,
            shuffle=False,
            preprocess_fn=self.preprocess,
        )

    @property
    def pilots(self):
        """Returns the pilot sequences"""
        raise NotImplementedError

    @property
    def num_symbols(self):
        raise NotImplementedError

    @property
    def num_subcarries(self):
        raise NotImplementedError

    @property
    def num_pilot_symbols(self):
        raise NotImplementedError

    @property
    def num_pilot_subcarriers(self):
        raise NotImplementedError
