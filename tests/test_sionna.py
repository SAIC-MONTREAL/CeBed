import pytest
import tensorflow as tf
import numpy as np
from cebed.datasets.sionna import (
    MultiDomainOfflineSionnaDataset,
    OfflineSionnaDataset,
    preprocess_labels,
    preprocess_inputs,
)
from cebed.envs import OfdmEnv


@pytest.fixture
def mock_datasets():
    return {"sd": "./tests/data/siso_data_sd", "md": "./tests/data/siso_data_md"}


def get_mock_mask(shape, input_type):

    if input_type != "low":
        return None

    mask = np.zeros(shape, bool)
    mask[..., 0, :] = True

    return mask.astype(np.float32)


def test_offline_dataset_setup(mock_datasets):
    dataset = OfflineSionnaDataset(path=mock_datasets["sd"])

    # Test size
    assert dataset.size == 100
    assert dataset.test_x.shape[0] == int(100 * 0.1)
    assert dataset.train_x.shape[0] == int(100 * 0.9 * 0.9)
    assert dataset.val_x.shape[0] == int(100 * 0.9 * 0.1)

    # dtype
    assert dataset.train_x.dtype == np.complex64
    assert dataset.train_y.dtype == np.complex64
    assert dataset.val_x.dtype == np.complex64
    assert dataset.val_y.dtype == np.complex64
    assert dataset.test_x.dtype == np.complex64
    assert dataset.test_y.dtype == np.complex64

    assert isinstance(dataset.env, OfdmEnv)

    # expected shape
    assert dataset.train_x.shape == (81, 1, 1, 1, 1, 14, 72)
    assert dataset.train_y.shape == (81, 1, 1, 1, 1, 14, 72)
    assert dataset.val_x.shape == (9, 1, 1, 1, 1, 14, 72)
    assert dataset.val_y.shape == (9, 1, 1, 1, 1, 14, 72)
    assert dataset.test_x.shape == (10, 1, 1, 1, 1, 14, 72)
    assert dataset.test_y.shape == (10, 1, 1, 1, 1, 14, 72)

    # mask
    assert dataset.mask is not None
    env = dataset.env
    assert dataset.mask.shape == [
        env.config.n_ues,
        env.config.num_streams_per_tx,
        env.config.num_ofdm_symbols,
        env.config.fft_size,
    ]
    assert dataset.mask.dtype == tf.float32

    assert (
        np.unique(np.where(dataset.inputs != 0)[-2])
        == dataset.env.pilot_ofdm_symbol_indices
    ).all()
    assert (
        len(np.unique(np.where(dataset.inputs != 0)[-2])) == dataset.env.n_pilot_symbols
    )
    assert (
        len(np.unique(np.where(dataset.inputs != 0)[-1]))
        == dataset.env.n_pilot_subcarriers
    )


@pytest.mark.parametrize("input_type", ["low", "raw"])
def test_offline_dataset_loaders(mock_datasets, input_type):
    dataset = OfflineSionnaDataset(path=mock_datasets["sd"], input_type=input_type)

    train_loader, val_loader = dataset.get_loaders(
        train_batch_size=32, eval_batch_size=32
    )

    batch = next(iter(train_loader))

    if input_type == "low":
        assert batch[0].shape == [32, 2, 72, 2]
        assert np.count_nonzero(batch[0]) == 32 * 2 * 72 * 2
    else:
        assert batch[0].shape == [32, 14, 72, 2]
        assert np.count_nonzero(batch[0]) == 32 * 2 * 72 * 2

    assert batch[1].shape == [32, 14, 72, 2]
    assert batch[0].dtype == tf.float32
    assert batch[1].dtype == tf.float32


def test_offline_multi_domain_dataset_setup(mock_datasets):
    dataset = MultiDomainOfflineSionnaDataset(path=mock_datasets["md"])

    # Test size
    assert dataset.size == 100 * 5
    assert dataset.num_domains == 5

    # dtype
    assert dataset.train_x.dtype == np.complex64
    assert dataset.train_y.dtype == np.complex64
    assert dataset.val_x.dtype == np.complex64
    assert dataset.val_y.dtype == np.complex64
    assert dataset.test_x.dtype == np.complex64
    assert dataset.test_y.dtype == np.complex64

    assert isinstance(dataset.env, OfdmEnv)

    # expected shape
    assert dataset.train_x.shape == (5, 81, 1, 1, 1, 1, 14, 72)
    assert dataset.train_y.shape == (5, 81, 1, 1, 1, 1, 14, 72)
    assert dataset.val_x.shape == (5, 9, 1, 1, 1, 1, 14, 72)
    assert dataset.val_y.shape == (5, 9, 1, 1, 1, 1, 14, 72)
    assert dataset.test_x.shape == (5, 10, 1, 1, 1, 1, 14, 72)
    assert dataset.test_y.shape == (5, 10, 1, 1, 1, 1, 14, 72)


@pytest.mark.parametrize("input_type", ["low", "raw"])
def test_offline_multi_domain_dataset_loaders(mock_datasets, input_type):
    dataset = MultiDomainOfflineSionnaDataset(
        path=mock_datasets["md"], input_type=input_type
    )

    train_loader, val_loader = dataset.get_loaders(
        train_batch_size=32, eval_batch_size=32
    )

    batch = next(iter(train_loader))

    if input_type == "low":
        assert batch[0].shape == [32, 2, 72, 2]
        assert np.count_nonzero(batch[0]) == 32 * 2 * 72 * 2
    else:
        assert batch[0].shape == [32, 14, 72, 2]
        assert np.count_nonzero(batch[0]) == 32 * 2 * 72 * 2

    assert batch[1].shape == [32, 14, 72, 2]
    assert batch[0].dtype == tf.float32
    assert batch[1].dtype == tf.float32


@pytest.mark.parametrize("nr", [1, 4])
@pytest.mark.parametrize("nu", [1, 2])
def test_preprocess_labels(nr, nu):
    shape = [1, 1, nr, nu, 1, 14, 72]
    labels = tf.complex(
        np.ones(shape, dtype=np.float32), -1 * np.ones(shape, dtype=np.float32)
    )

    with pytest.raises(
        ValueError, match="Input shape can to have more than 6 dimensions"
    ):
        preprocess_labels(labels)

    shape = [1, nr, nu, 1, 14, 72]
    labels = tf.complex(
        np.ones(shape, dtype=np.float32), -1 * np.ones(shape, dtype=np.float32)
    )

    pre_labels = preprocess_labels(labels)

    assert pre_labels.shape == [14, 72, nr * nu * 2]
    assert pre_labels.dtype == tf.float32
    assert (pre_labels[:, :, : nr * nu].numpy() == np.ones([14, 72, nr * nu])).all()
    assert (
        pre_labels[:, :, nr * nu :].numpy() == -1 * np.ones([14, 72, nr * nu])
    ).all()


@pytest.mark.parametrize("input_type", ["low", "raw"])
@pytest.mark.parametrize("nr", [1, 4])
@pytest.mark.parametrize("nu", [1, 2])
def test_preprocess_inputs(input_type, nr, nu):

    shape = [1, 1, nr, nu, 1, 14, 72]
    inputs = tf.complex(
        np.ones(shape, dtype=np.float32), -1 * np.ones(shape, dtype=np.float32)
    )

    mask = get_mock_mask(shape, input_type)

    with pytest.raises(
        ValueError, match="Input shape can to have more than 6 dimensions"
    ):
        preprocess_inputs(inputs, input_type=input_type, mask=mask)

    shape = [1, nr, nu, 1, 14, 72]
    inputs = tf.complex(
        np.ones(shape, dtype=np.float32), -1 * np.ones(shape, dtype=np.float32)
    )
    mask = get_mock_mask(shape, input_type)
    pre_inputs = preprocess_inputs(inputs, input_type=input_type, mask=mask)

    if input_type == "raw":
        h = 14
    else:
        h = 1
    assert pre_inputs.shape == [h, 72, nr * nu * 2]
    assert pre_inputs.dtype == tf.float32

    assert (pre_inputs[:, :, : nr * nu].numpy() == 1 * np.ones([h, 72, nr * nu])).all()
    assert (pre_inputs[:, :, nr * nu :].numpy() == -1 * np.ones([h, 72, nr * nu])).all()


def test_get_input_shape(mock_datasets):
    dataset = OfflineSionnaDataset(path=mock_datasets["sd"], input_type="random")

    with pytest.raises(ValueError, match="Unknown input type random"):
        _ = dataset.get_input_shape()
