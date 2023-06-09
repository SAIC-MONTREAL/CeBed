"""
Default tests for Env classes
"""
import pytest
import numpy as np
import tensorflow as tf

from sionna.ofdm import PilotPattern
from cebed.envs import OfdmEnv, EnvConfig


def mock_pilot_pattern(config):
    """Dummy pilot pattern where the pilots are set to one"""
    shape = [
        config.n_ues,
        config.num_streams_per_tx,
        config.num_ofdm_symbols,
        config.fft_size,
    ]

    mask = np.zeros(shape, bool)
    mask[..., 3, :] = True
    shape[2] = 1
    pilots = np.zeros(shape, np.complex64)
    pilots[..., 0, :] = np.ones((config.fft_size,), np.complex64)

    pilots = np.reshape(pilots, [config.n_ues, config.num_streams_per_tx, -1])

    return PilotPattern(mask=mask, pilots=pilots)


@pytest.mark.parametrize("n_ues", [1, 4])
@pytest.mark.parametrize("nr", [1, 4])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_env(n_ues, nr):
    """test env works properly"""
    config = EnvConfig()
    config.num_rx_antennas = nr
    config.n_ues = n_ues
    env = OfdmEnv(config)

    batch_size = 10
    snr_db = 20

    outputs = env(batch_size, snr_db)

    assert len(outputs) == 2
    expected_y_shape = [
        batch_size,
        1,
        config.num_rx_antennas,
        config.num_ofdm_symbols,
        config.fft_size,
    ]
    expected_h_shape = [
        batch_size,
        1,
        config.num_rx_antennas,
        config.n_ues,
        config.num_streams_per_tx,
        config.num_ofdm_symbols,
        config.fft_size,
    ]

    assert outputs[0].shape == expected_y_shape
    assert outputs[1].shape == expected_h_shape

    outputs = env(batch_size, snr_db, return_x=True)

    assert len(outputs) == 3
    expected_x_shape = [
        batch_size,
        config.n_ues,
        config.num_streams_per_tx,
        config.num_ofdm_symbols,
        config.fft_size,
    ]
    assert outputs[0].shape == expected_x_shape


@pytest.mark.parametrize("p_spacing", [1, 2])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_block_pilot_pattern_values(p_spacing):
    """Block pilot pattern values"""
    config = EnvConfig()
    config.p_spacing = p_spacing
    env = OfdmEnv(config)

    for i in range(0, config.num_ofdm_symbols):
        if i not in env.pilot_ofdm_symbol_indices:
            print(env.get_mask().shape)
            assert all(env.get_mask()[0, 0, i] == tf.zeros(shape=(config.fft_size,)))

    indices = np.arange(0, config.fft_size, config.p_spacing)

    for i in env.pilot_ofdm_symbol_indices:
        for j in indices:
            assert env.get_mask()[0, 0, i, j] == 1


@pytest.mark.parametrize("nues", [2, 4])
def test_get_mask(nues):
    config = EnvConfig()
    config.n_ues = nues
    env = OfdmEnv(config)

    mask = env.get_mask()
    assert mask.shape == [
        nues,
        env.config.num_streams_per_tx,
        env.config.num_ofdm_symbols,
        env.config.fft_size,
    ]


@pytest.mark.parametrize("p_spacing", [1, 2])
@pytest.mark.parametrize("nr", [4, 8])
@pytest.mark.parametrize("nues", [2, 4])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_mimo_block_pilot_pattern(p_spacing, nr, nues):
    """Test block pilot pattern properties"""
    config = EnvConfig()
    config.num_rx_antennas = nr
    config.n_ues = nues
    config.p_spacing = p_spacing
    env = OfdmEnv(config)

    assert env.n_pilot_symbols == len(config.pilot_ofdm_symbol_indices)
    assert env.n_pilot_subcarriers == int(
        env.rg.num_effective_subcarriers / config.p_spacing
    )

    mask = env.get_mask()

    assert int(np.count_nonzero(mask)) / nues == env.rg.num_pilot_symbols.numpy()


def test_extract_at_pilot_locations():
    """test extract at pilot locations"""
    config = EnvConfig()

    config.pilot_pattern = mock_pilot_pattern(config)

    env = OfdmEnv(config)

    batch_size = 10

    y_shape = [
        batch_size,
        1,
        config.num_rx_antennas,
        config.num_ofdm_symbols,
        config.fft_size,
    ]

    y = np.ones(y_shape, dtype=np.complex64)
    y[:, 0, :, 3, :] = -1 * np.ones((config.fft_size,))

    yp = env.extract_at_pilot_locations(y)

    expect_yp_shape = [
        batch_size,
        1,
        config.num_rx_antennas,
        config.n_ues,
        config.num_streams_per_tx,
        env.rg.pilot_pattern.num_pilot_symbols.numpy(),
    ]

    assert yp.shape == expect_yp_shape

    assert (yp.numpy() == -1 * np.ones(expect_yp_shape, np.complex64)).all()

    h_hat = env.estimate_at_pilot_locations(y)

    expected_h_shape = [
        batch_size,
        1,
        config.num_rx_antennas,
        config.n_ues,
        config.num_streams_per_tx,
        config.num_ofdm_symbols,
        config.fft_size,
    ]

    assert h_hat.shape == expected_h_shape

    assert (
        h_hat[:, 0, :, :, 0, 3, :].numpy()
        == -1 * np.ones((config.fft_size,), np.complex64)
    ).all()

    for i in range(config.num_ofdm_symbols):
        if i != 3:
            assert (
                h_hat[:, 0, :, :, 0, i, :].numpy()
                == np.zeros((config.fft_size,), np.complex64)
            ).all()
