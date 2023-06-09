"""Eval helper functions"""
import os
from typing import Tuple
import tensorflow as tf
import numpy as np

from cebed.envs import OfdmEnv


def mse(true: tf.Tensor, pred: tf.Tensor) -> float:
    """Computes mean squared error"""

    return tf.reduce_mean(tf.square(tf.abs(true - pred)))


def estimate_covariance_matrices(
    env: OfdmEnv, num_it: int = 10, batch_size: int = 1000, save_dir: str = "./eval"
) -> Tuple[tf.Tensor]:
    """
    Estimates the second order statistics of the channel

    Taken from
    https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#

    """

    if os.path.isfile(f"{save_dir}/freq_cov_mat"):
        freq_cov_mat = np.load(f"{save_dir}/freq_cov_mat.npy")
        time_cov_mat = np.load(f"{save_dir}/time_cov_mat.npy")
        space_cov_mat = np.load(f"{save_dir}/space_cov_mat.npy")
        freq_cov_mat = tf.constant(freq_cov_mat, tf.complex64)
        time_cov_mat = tf.constant(time_cov_mat, tf.complex64)
        space_cov_mat = tf.constant(space_cov_mat, tf.complex64)

        return freq_cov_mat, time_cov_mat, space_cov_mat

    freq_cov_mat = tf.zeros([env.config.fft_size, env.config.fft_size], tf.complex64)
    time_cov_mat = tf.zeros(
        [env.config.num_ofdm_symbols, env.config.num_ofdm_symbols], tf.complex64
    )
    space_cov_mat = tf.zeros(
        [env.config.num_rx_antennas, env.config.num_rx_antennas], tf.complex64
    )

    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = env.sample_channel(batch_size)

        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0, 2, 1, 3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0, 1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(
        tf.cast(env.config.num_ofdm_symbols * num_it, tf.float32), 0.0
    )
    time_cov_mat /= tf.complex(tf.cast(env.config.fft_size * num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(env.config.fft_size * num_it, tf.float32), 0.0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/freq_cov_mat", freq_cov_mat.numpy())
    np.save(f"{save_dir}/time_cov_mat", time_cov_mat.numpy())
    np.save(f"{save_dir}/space_cov_mat", space_cov_mat.numpy())

    return freq_cov_mat, time_cov_mat, space_cov_mat
