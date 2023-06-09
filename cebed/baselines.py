"""Implements several channel estimation baselines"""

from typing import Tuple, List
import numpy as np
import tensorflow as tf

from cebed.utils import tfinterpolate


def linear_interpolation(
    mat, new_shape: Tuple[int], method: str = "bilinear"
) -> tf.Tensor:
    """
    Upscale a matrice to a given shape using linear interpolation
    For complex inputs, the real and imaginary parts are interpolated seperatly
    """
    mat_interpolated = tfinterpolate(mat, new_shape, method=method)

    return mat_interpolated


def lmmse_interpolation(h_p, h, h_ls, snr_db, pilot_locations):
    """
    LMMSE interpolation
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4699619
    """

    shape = tf.shape(h_ls)
    n_rx_ant = shape[2]
    n_tx = shape[3]
    h_p = tf.reshape(h_p, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1]))
    h_ls = tf.reshape(h_ls, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1]))
    h = tf.reshape(h, (shape[0], n_rx_ant, n_tx, h.shape[-2], h.shape[-1]))

    # [batch, nr, nt, n_pilot_symbols, n_subcarrriers]
    h_mmse = np.zeros((shape[0], n_rx_ant, n_tx, shape[-2], h.shape[-1])).astype(
        complex
    )

    for b in range(shape[0]):
        for nt in range(n_tx):
            for nr in range(n_rx_ant):
                for k, i in enumerate(pilot_locations):
                    hp = h_p[b, nr, nt, k, :].numpy()[:, None]
                    hi = h[b, nr, nt, i, :].numpy()[:, None]
                    rhp = np.matmul(hp, hp.conj().T)
                    rhhp = np.matmul(hi, hp.conj().T)
                    pinv = np.linalg.pinv(
                        rhp + (1 / (10.0 ** (snr_db / 10.0))) * np.eye(rhp.shape[0])
                    )
                    a = np.matmul(rhhp, pinv)
                    h_mmse[b, nr, nt, k, :] = np.matmul(a, h_ls[b, nr, nt, k, :])

    return h_mmse


def linear_ls_baseline(
    h_ls: tf.Tensor, num_ofdm_symbols: int, num_ofdm_subcarriers: int
):
    """Linear interpolation of the LS estimates"""
    # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilot_symbols, num_pilot_subcarriers]

    shape = tf.shape(h_ls)
    n_rx_ant = shape[2]
    n_tx = shape[3]
    # [num_batch, nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_ls_r = tf.reshape(h_ls, (shape[0], n_rx_ant * n_tx, shape[-2], shape[-1]))
    # [num_batch, num_pilot_symbols,num_pilot_subcarriers, nrx_ant*nt]
    h_ls_r = tf.transpose(h_ls_r, [0, 2, 3, 1])
    # [num_batch, num_symbols,num_subcarriers, nrx_ant*nt]
    h_ls_lin = linear_interpolation(h_ls_r, (num_ofdm_symbols, num_ofdm_subcarriers))
    # [num_batch, , nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_ls_lin = tf.transpose(h_ls_lin, [0, 3, 1, 2])
    # [num_batch, , nrx_ant, nt, num_pilot_symbols,num_pilot_subcarriers]
    shape = tf.shape(h_ls_lin)
    h_ls_lin = tf.reshape(h_ls_lin, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1]))

    return h_ls_lin


def lmmse_baseline(
    h_p: tf.Tensor,
    h_freq: tf.Tensor,
    h_ls: tf.Tensor,
    snr_db: int,
    pilot_locations: List[int],
    num_ofdm_symbols: int,
    num_ofdm_subcarriers: int,
) -> tf.Tensor:
    """
    Idea LMMSE baseline.
    :param h_p: The LS estimates at pilot positions of noise free channel
    :param h_freq: The true channel coefficients
    :param h_ls: The LS estimates at pilot positions of noisy channel
    :param snr_db: The noise level in dB
    :param pilot_locations: A list of the pilot symbol indices
    :param num_ofdm_symbols: The number of symbols in the resource grid
    :param num_ofdm_subcarriers: The number of subcarriers in the resource grid

    :return The LMMSE estimates

    """

    # [num_batch, nrx_ant, nt, num_pilot_symbols,num_pilot_subcarriers]
    h_lmmse = lmmse_interpolation(h_p, h_freq, h_ls, snr_db, pilot_locations)

    shape = tf.shape(h_lmmse)
    n_rx_ant = shape[1]
    n_tx = shape[2]
    # [num_batch, nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_lmmse_r = tf.reshape(h_lmmse, (shape[0], n_rx_ant * n_tx, shape[-2], shape[-1]))
    # [num_batch, num_pilot_symbols,num_pilot_subcarriers, nrx_ant*nt]
    h_lmmse_r = tf.transpose(h_lmmse_r, [0, 2, 3, 1])
    # [num_batch, num_symbols,num_subcarriers, nrx_ant*nt]
    h_lmmse_frame = linear_interpolation(
        h_lmmse_r, (num_ofdm_symbols, num_ofdm_subcarriers)
    )
    # [num_batch, , nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_lmmse_frame = tf.transpose(h_lmmse_frame, [0, 3, 1, 2])
    # [num_batch, , nrx_ant, nt, num_pilot_symbols,num_pilot_subcarriers]
    shape = tf.shape(h_lmmse_frame)
    h_lmmse_frame = tf.reshape(
        h_lmmse_frame, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1])
    )

    return h_lmmse_frame
