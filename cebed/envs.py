"""
OFDM Environment based on Sionna
"""

import itertools
from typing import List, Any, Union, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

import tensorflow as tf

from sionna.utils import QAMSource
from sionna.ofdm import ResourceGrid, ResourceGridMapper, PilotPattern
from sionna.channel import (
    GenerateOFDMChannel,
    OFDMChannel,
    gen_single_sector_topology,
    RayleighBlockFading,
)
from sionna.channel.tr38901 import UMi, Antenna, PanelArray, UMa, RMa
from sionna.ofdm import RemoveNulledSubcarriers
from sionna.utils import flatten_last_dims


@dataclass
class EnvConfig:
    """Configuration for OfdmEnv class"""

    # Number of UEs
    n_ues: int = 1
    # Carrier frequency
    carrier_frequency: int = 3e9
    # Number of OFDM symbols in one frame
    num_ofdm_symbols: int = 14
    # Number of subcarriers i one frame
    fft_size: int = 72
    # The subcarrier subspacing
    subcarrier_spacing: int = 30e3
    # Number of antennas at the receiver
    num_rx_antennas: int = 1
    # The channel model
    scenario: str = "umi"
    # The number of bits per symbols for the modulation
    num_bits_per_symbol: int = 2
    # The pilot pattern ["block", "kronecker"]
    pilot_pattern: str = "block"
    # The positions of the pilot symbols
    pilot_ofdm_symbol_indices: List[int] = field(default_factory=lambda: [3, 10])
    # The pilot spacing between subcarriers
    p_spacing: int = 2
    # Whether to include path loss or not
    path_loss: bool = False
    # Wether to include shadowing or nor
    shadowing: bool = False
    # Wether there is a line of sight or not
    los: bool = False
    # The tranmission direction
    direction: str = "uplink"
    # The number of streams by transmitter
    num_streams_per_tx: int = 1
    # The user speed
    ue_speed: int = 3
    # Wether to change the topology at each call
    dynamic_topology: int = True
    # Wether to normalize the channels or not
    normalize_channel: int = True
    # The seed
    seed = 0

    def from_dict(self, kwargs: Dict[str, Any]) -> None:
        """Update config from dict"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def create_channel_model(
    config: EnvConfig, ue_antenna: Antenna, bs_array: PanelArray
) -> Any:
    """
    Create the channel model given the user-config

    :param config: The environment configuration
    :param ue_antenna: The antenna pattern of the UE
    :param bs_array: The antenna array at the base station

    :return A channel model
    """

    if config.scenario == "umi":
        return UMi(
            carrier_frequency=config.carrier_frequency,
            o2i_model="low",
            ut_array=ue_antenna,
            bs_array=bs_array,
            direction=config.direction,
            enable_shadow_fading=config.shadowing,
            enable_pathloss=config.path_loss,
        )

    if config.scenario == "uma":
        return UMa(
            carrier_frequency=config.carrier_frequency,
            o2i_model="low",
            ut_array=ue_antenna,
            bs_array=bs_array,
            direction=config.direction,
            enable_shadow_fading=config.shadowing,
            enable_pathloss=config.path_loss,
        )

    if config.scenario == "rma":
        return RMa(
            carrier_frequency=config.carrier_frequency,
            ut_array=ue_antenna,
            bs_array=bs_array,
            direction=config.direction,
            enable_shadow_fading=config.shadowing,
            enable_pathloss=config.path_loss,
        )

    if config.scenario == "Rayleigh":
        return RayleighBlockFading(
            num_rx=1,  # 1 BS
            num_rx_ant=config.num_rx_antennas,
            num_tx=config.n_ues,
            num_tx_ant=1,
        )
    raise ValueError(f"Unsupported channel model {config.scenario}")


class OfdmEnv:
    """
    Define an OFDM system
    """

    def __init__(self, config: EnvConfig):
        self.config = config

        # The UEs are equipped with a single antenna with vertial polarization.
        ue_antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",  # Omnidirectional antenna pattern
            carrier_frequency=config.carrier_frequency,
        )

        if config.num_rx_antennas == 1:
            bs_array = Antenna(
                polarization="single",
                polarization_type="V",
                antenna_pattern="omni",  # Omnidirectional antenna pattern
                carrier_frequency=config.carrier_frequency,
            )
        else:
            # Dual Pol ULA array
            bs_array = PanelArray(
                num_cols_per_panel=int(config.num_rx_antennas / 2),
                num_rows_per_panel=1,
                polarization="dual",
                polarization_type="cross",
                antenna_pattern="38.901",  # 3GPP 38.901 antenna pattern
                carrier_frequency=config.carrier_frequency,
            )

        self.channel_model = create_channel_model(config, ue_antenna, bs_array)

        # Pilot pattern
        _pilot_pattern = self.create_pilot_pattern()

        # Resource Grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=config.num_ofdm_symbols,
            fft_size=config.fft_size,
            subcarrier_spacing=config.subcarrier_spacing,
            num_tx=config.n_ues,
            pilot_pattern=_pilot_pattern,
            num_streams_per_tx=config.num_streams_per_tx,
            pilot_ofdm_symbol_indices=config.pilot_ofdm_symbol_indices,
        )

        ##################################
        # Transmitter
        ##################################
        self.qam_source = QAMSource(num_bits_per_symbol=config.num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)

        ##################################
        # Channel
        ##################################
        self.channel = OFDMChannel(
            channel_model=self.channel_model,
            resource_grid=self.rg,
            add_awgn=True,
            normalize_channel=config.normalize_channel,
            return_channel=True,
        )

    @property
    def pilot_ofdm_symbol_indices(self):
        """Pilot symbols positions in the grid"""

        return self.config.pilot_ofdm_symbol_indices

    @property
    def n_pilot_symbols(self):
        """Number of pilot symbols"""

        return len(self.pilot_ofdm_symbol_indices)

    @property
    def n_pilot_subcarriers(self):
        """Number of pilot subcarriers"""

        if self.config.pilot_pattern == "kronecker":
            return self.rg.num_effective_subcarriers

        if self.config.pilot_pattern == "block":
            return int(self.rg.num_effective_subcarriers / self.config.p_spacing)
        raise ValueError("Unrecognized pilot pattern")

    def create_pilot_pattern(self):
        """
        Create the pilot pattern
        """

        if self.config.pilot_pattern == "block":
            return self.block_pilot_pattern(spacing=self.config.p_spacing)

        return self.config.pilot_pattern

    def set_new_topology(self, batch_size: int):
        """Set new topology"""

        if self.config.scenario in ["umi", "uma", "rma"]:
            topology = gen_single_sector_topology(
                batch_size,
                self.config.n_ues,
                max_ut_velocity=self.config.ue_speed,
                min_ut_velocity=self.config.ue_speed,
                scenario=self.config.scenario,
            )
            self.channel_model.set_topology(*topology, los=self.config.los)

    def generate_symbols(self, batch_size: int) -> tf.Tensor:
        """Generate a batch of symbols to be transmitted"""
        symbols = self.qam_source(
            [
                batch_size,
                self.config.n_ues,
                self.config.num_streams_per_tx,
                self.rg.num_data_symbols,
            ]
        )

        return self.rg_mapper(symbols)

    def __call__(
        self, batch_size: int, snr_db: int, return_x: bool = False
    ) -> Union[Optional[tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Run the wireless system and return a batch of data
        :param batch_size: The batch size
        :param snr_db: The noise level
        :param: return_x: Wether to return the sent symbols or not

        :return x (optional), y, h
        """

        if self.config.dynamic_topology:
            self.set_new_topology(batch_size)

        ##################################
        # Transmitter
        ##################################

        x_rg = self.generate_symbols(batch_size)

        ##################################
        # Channel
        ##################################

        noise = tf.pow(10.0, -snr_db / 10.0)

        received_signal, channel = self.channel((x_rg, noise))

        if return_x:
            return x_rg, received_signal, channel

        return received_signal, channel

    def get_mask(self) -> tf.Tensor:
        """
        Return the pilot mask.
        The pilot positions are 1 and data positions are 0.
        """

        return tf.cast(self.rg.pilot_pattern.mask, tf.float32)

    # tf.cast(tf.squeeze(self.rg.pilot_pattern.mask), tf.float32)

    def sample_channel(self, batch_size: int, squeeze: bool = True) -> tf.Tensor:
        """Generate a batch of samples using the channel model
        :param batch_size: The batch size
        :param squeeze: If True, some dimensions will be removed
        """

        channel_sampler = GenerateOFDMChannel(
            self.channel_model, self.rg, normalize_channel=True
        )

        self.set_new_topology(batch_size)

        # Sample channel frequency responses
        # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
        h_freq = channel_sampler(batch_size)

        if squeeze:
            # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
            h_freq = h_freq[:, 0, :, 0, 0]

        return h_freq

    def extract_at_pilot_locations(self, y: tf.Tensor) -> tf.Tensor:
        """
        Extract the pilot symbols from the received signal y
        Based on the code from
        https://nvlabs.github.io/sionna/_modules/sionna/ofdm/channel_estimation.html#LSChannelEstimator

        :param y: The received signal
        """
        y_eff = RemoveNulledSubcarriers(self.rg)(y)
        y_eff_flat = flatten_last_dims(y_eff)
        mask = flatten_last_dims(self.rg.pilot_pattern.mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
        num_pilot_symbols = self.rg._pilot_pattern.num_pilot_symbols
        _pilot_ind = pilot_ind[..., :num_pilot_symbols]
        y_pilots = tf.gather(y_eff_flat, _pilot_ind, axis=-1)

        return y_pilots

    def estimate_at_pilot_locations(self, y: tf.Tensor) -> tf.Tensor:
        """
        Estimates the channel for the pilot-carrying resource elements.
        Based on the code from
        https://nvlabs.github.io/sionna/_modules/sionna/ofdm/channel_estimation.html#LSChannelEstimator
        :param y: The received signal

        """

        y_pilots = self.extract_at_pilot_locations(y)

        h_ls = tf.math.divide_no_nan(y_pilots, self.rg._pilot_pattern.pilots)

        # Build pilot masks for every stream
        pilot_mask = self._build_pilot_mask()

        # Build indices for mapping channel estimates and
        # that are given as input to a resource grid
        num_pilots = self.rg.pilot_pattern.pilots.shape[2]
        inputs_to_rg_indices = self._build_inputs2rg_indices(pilot_mask, num_pilots)
        _inputs_to_rg_indices = tf.cast(inputs_to_rg_indices, tf.int32)

        batch_size = tf.shape(h_ls)[0]
        num_rx = tf.shape(h_ls)[1]
        num_rx_ant = tf.shape(h_ls)[2]
        num_tx = tf.shape(h_ls)[3]
        num_tx_stream = tf.shape(h_ls)[4]
        num_ofdm_symbols = self.rg.pilot_pattern.num_ofdm_symbols
        num_effective_subcarriers = self.rg.pilot_pattern.num_effective_subcarriers
        h_hat = tf.transpose(h_ls, [3, 4, 5, 0, 1, 2])
        h_hat = tf.scatter_nd(
            _inputs_to_rg_indices,
            h_hat,
            [
                num_tx,
                num_tx_stream,
                num_ofdm_symbols,
                num_effective_subcarriers,
                batch_size,
                num_rx,
                num_rx_ant,
            ],
        )
        h_hat = tf.transpose(h_hat, [4, 5, 6, 0, 1, 2, 3])

        return h_hat

    def _build_inputs2rg_indices(
        self, pilot_mask: tf.Tensor, num_pilots: int
    ) -> tf.Tensor:
        """
        Builds indices for mapping channel estimates
        that are given as input to a resource grid
        source:
        https://nvlabs.github.io/sionna/_modules/sionna/ofdm/channel_estimation.html
        """

        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]
        num_ofdm_symbols = pilot_mask.shape[2]
        num_effective_subcarriers = pilot_mask.shape[3]

        inputs_to_rg_indices = np.zeros(
            [num_tx, num_streams_per_tx, num_pilots, 4], int
        )

        for tx, st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0  # Pilot index for this stream

            for sb, sc in itertools.product(
                range(num_ofdm_symbols), range(num_effective_subcarriers)
            ):

                if pilot_mask[tx, st, sb, sc] == 0:
                    continue

                if pilot_mask[tx, st, sb, sc] == 1:
                    inputs_to_rg_indices[tx, st, pil_index] = [tx, st, sb, sc]
                pil_index += 1

        return inputs_to_rg_indices

    def _build_pilot_mask(self) -> tf.Tensor:
        """
        Build for every transmitter and stream a pilot mask indicating
        which REs are allocated to pilots, data, or not used.
        # 0 -> Data
        # 1 -> Pilot
        # 2 -> Not used
        """

        pilot_pattern = self.rg.pilot_pattern
        mask = pilot_pattern.mask
        pilots = pilot_pattern.pilots
        num_tx = mask.shape[0]
        num_streams_per_tx = mask.shape[1]
        num_ofdm_symbols = mask.shape[2]
        num_effective_subcarriers = mask.shape[3]

        pilot_mask = np.zeros(
            [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers],
            int,
        )

        for tx, st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0

            for sb, sc in itertools.product(
                range(num_ofdm_symbols), range(num_effective_subcarriers)
            ):

                if mask[tx, st, sb, sc] == 1:

                    if tf.abs(pilots[tx, st, pil_index]) > 0.0:
                        pilot_mask[tx, st, sb, sc] = 1
                    else:
                        pilot_mask[tx, st, sb, sc] = 2
                    pil_index += 1

        return pilot_mask

    def block_pilot_pattern(self, spacing=2) -> PilotPattern:
        """
        Create a block pilot pattern
        # TODO this assumes no dc nor guard subcarriers are used
        :param p_spacing: The spacing between pilot subcarriers
        """
        num_pilot_symbols = len(self.pilot_ofdm_symbol_indices)

        num_pilot_subc = int(self.config.fft_size / spacing)

        num_seq = self.config.n_ues * self.config.num_streams_per_tx

        num_pilots = num_pilot_symbols * num_pilot_subc / num_seq
        assert (
            num_pilots % 1 == 0
        ), """`num_effective_subcarriers` must be an integer multiple of `num_tx`*`num_streams_per_tx`."""

        # Number of pilots per OFDM symbol
        num_pilots_per_symbol = int(num_pilots / num_pilot_symbols)
        # Prepare empty mask and pilots
        shape = [
            self.config.n_ues,
            self.config.num_streams_per_tx,
            self.config.num_ofdm_symbols,
            self.config.fft_size,
        ]
        mask = np.zeros(shape, bool)
        shape[2] = num_pilot_symbols
        shape[-1] = num_pilot_subc
        pilots = np.zeros(shape, np.complex64)

        # Populate all the selected OFDM symbols in the mask
        pilot_locations = np.arange(0, self.config.fft_size, spacing)
        subcarrier_ids = np.tile(pilot_locations, len(self.pilot_ofdm_symbol_indices))
        symbol_ids = np.repeat(self.pilot_ofdm_symbol_indices, len(pilot_locations))

        mask[..., symbol_ids, subcarrier_ids] = True

        # Populate the pilots with random QPSK symbols
        qam_source = QAMSource(2, seed=self.config.seed, dtype=tf.complex64)

        for i in range(self.config.n_ues):
            for j in range(self.config.num_streams_per_tx):
                p = qam_source([1, 1, num_pilot_symbols, num_pilots_per_symbol])
                pilots[i, j, :, i * self.config.num_streams_per_tx + j :: num_seq] = p

        pilots = np.reshape(
            pilots, [self.config.n_ues, self.config.num_streams_per_tx, -1]
        )

        return PilotPattern(
            mask, pilots, trainable=False, normalize=True, dtype=tf.complex64
        )
