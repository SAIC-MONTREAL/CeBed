"""Test eval helper functions"""

import pytest
import tensorflow as tf
from cebed.evaluation import estimate_covariance_matrices
from cebed.envs import OfdmEnv, EnvConfig


def test_estimate_covariance_matrices():
    """test covariance estimation"""
    config = EnvConfig()
    env = OfdmEnv(config)

    freq_cov_mat, time_cov_mat, space_cov_mat = estimate_covariance_matrices(
        env, num_it=1, batch_size=5, save_dir="./eval/cov_mat"
    )

    assert freq_cov_mat.dtype == tf.complex64
    assert time_cov_mat.dtype == tf.complex64
    assert space_cov_mat.dtype == tf.complex64

    assert freq_cov_mat.shape == (env.config.fft_size, env.config.fft_size)
    assert time_cov_mat.shape == (
        env.config.num_ofdm_symbols,
        env.config.num_ofdm_symbols,
    )
    assert space_cov_mat.shape == (
        env.config.num_rx_antennas,
        env.config.num_rx_antennas,
    )
