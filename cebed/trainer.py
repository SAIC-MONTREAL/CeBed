"""
Main code to train a  model
"""
import os
from typing import List, Dict
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
from sionna.channel import ApplyOFDMChannel
from sionna.ofdm import LSChannelEstimator, LMMSEInterpolator

# from cebed.datasets.sionna import MultiDomainDataset
import cebed.datasets as cds
import cebed.models as cm
from cebed.baselines import linear_ls_baseline, lmmse_baseline
from cebed.utils import unflatten_last_dim, write_metadata
from cebed.datasets.utils import postprocess
from cebed.evaluation import mse, estimate_covariance_matrices
from cebed.envs import OfdmEnv


@dataclass
class TrainConfig:
    """Train config"""

    experiment_name: str = "SionnaSisoMD"
    data_dir: str = "./datasets"
    dataset_name: str = "SionnaOfflineMD"

    model_name: str = "ReEsNet"
    seed: int = 42
    verbose: int = 1
    output_dir: str = "train_output"

    epochs: int = 100
    train_batch_size: int = 32
    eval_batch_size: int = 32
    lr: float = 0.001
    loss_fn: str = "mse"
    train_split: float = 0.9
    early_stopping: bool = False

    input_type: str = "low"


class Trainer:
    """
    Trainer class
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.log_dir = os.path.join(
            self.config.output_dir,
            self.config.experiment_name,
            str(self.config.seed),
            self.config.model_name,
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.train_loader = None
        self.eval_loader = None
        self.model = None
        self.dataset = None
        self.optimizer = tf.keras.optimizers.Adam(self.config.lr)
        self.loss_fn = self.config.loss_fn

    def setup(self):
        """Setup the trainer"""
        # Create datasets
        dataset_class = cds.get_dataset_class(self.config.dataset_name)
        self.dataset = dataset_class(
            self.config.data_dir,
            train_split=self.config.train_split,
            input_type=self.config.input_type,
            seed=self.config.seed,
        )
        self.train_loader, self.eval_loader = self.dataset.get_loaders(
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
        )

        # Create model
        model_hparams = cm.get_model_hparams(
            self.config.model_name, self.config.experiment_name  # self.config.data_dir
        )

        model_class = cm.get_model_class(self.config.model_name)
        if "output_dim" not in model_hparams:
            model_hparams["output_dim"] = self.dataset.output_shape

        self.model = model_class(model_hparams)

        input_shape = self.dataset.get_input_shape()
        self.model.build(
            tf.TensorShape([None, input_shape[0], input_shape[1], input_shape[2]])
        )

    def save(self):
        config = deepcopy(asdict(self.config))
        if (self.dataset.env is not None) and (hasattr(self.dataset.env, "config")):
            config.update(asdict(self.dataset.env.config))
        print(config)
        write_metadata(os.path.join(self.log_dir, "config.yaml"), config)

    def train(self):
        """Train the model"""
        callbacks = self.get_training_callbacks()

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        if self.config.verbose > 0:
            print(self.model.summary(expand_nested=True))
            print(
                f"Start training for \
                        seed {self.config.seed} \
                        with a learning rate {self.config.lr}"
            )
        start_time = time.time()
        self.model.fit(
            self.train_loader,
            verbose=self.config.verbose,
            epochs=self.config.epochs,
            validation_data=self.eval_loader,
            callbacks=callbacks,
        )

        print(f"Finished training in {time.time()-start_time:.2f}")

    def get_training_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Returns the training callbacks"""

        ckpt_folder = os.path.join(self.log_dir, "cp.ckpt")
        # Checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_folder,
            save_weights_only=True,
            verbose=self.config.verbose,
            monitor="val_loss",
            model="min",
            save_best_only=True,
        )

        # Tensorboard callback
        tensorboard_folder = os.path.join(self.log_dir, "tensorboard")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_folder
        )

        # history logging callback
        csv_logger_filename = os.path.join(self.log_dir, "csv_logger.csv")
        history_logger = tf.keras.callbacks.CSVLogger(
            csv_logger_filename, separator=",", append=True
        )

        # training callbakcs
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        )

        callbacks = [
            tensorboard_callback,
            checkpoint_callback,
            lr_callback,
            history_logger,
        ]

        if self.config.early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10
            )
            callbacks.append(es_callback)

        return callbacks

    def evaluate_ls(
        self, noisy_signals: tf.Tensor, interpolate: bool = True
    ) -> tf.Tensor:
        """LS method with bilinear interpolation"""

        y_pilot_noisy = self.dataset.env.extract_at_pilot_locations(noisy_signals)
        # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilots]
        h_ls = tf.math.divide_no_nan(y_pilot_noisy, self.dataset.pilots)
        # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilot_symbols, num_pilot_subcarriers]
        h_ls = unflatten_last_dim(
            h_ls, (self.dataset.num_pilot_symbols, self.dataset.num_pilot_subcarriers)
        )

        if not interpolate:
            return h_ls
        h_ls_lin = linear_ls_baseline(
            h_ls, self.dataset.num_symbols, self.dataset.num_subcarries
        )

        return h_ls_lin

    def evaluate_lmmse(
        self,
        noiseless_signals: tf.Tensor,
        channels: tf.Tensor,
        hls: tf.Tensor,
        snr_db: int,
    ) -> tf.Tensor:
        """Ideal LMMSE"""
        y_pilot_noise_free = self.dataset.env.extract_at_pilot_locations(
            noiseless_signals
        )
        # Channels at pilot positions noise free
        noiseless_hls = tf.math.divide_no_nan(y_pilot_noise_free, self.dataset.pilots)
        # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilot_symbols, num_pilot_subcarriers]
        noiseless_hls = unflatten_last_dim(
            noiseless_hls,
            (self.dataset.num_pilot_symbols, self.dataset.num_pilot_subcarriers),
        )
        h_lmmse = lmmse_baseline(
            noiseless_hls,
            channels,
            hls,
            snr_db,
            self.dataset.env.pilot_ofdm_symbol_indices,
            self.dataset.num_symbols,
            self.dataset.num_subcarries,
        )

        return h_lmmse

    def evaluate_almmse(self, noisy_signals: tf.Tensor, noise_lin: float) -> tf.Tensor:
        """ALMMSE baseline. Only work with Sionna environments"""

        freq_cov_mat, time_cov_mat, space_cov_mat = estimate_covariance_matrices(
            self.dataset.env, save_dir=f"{self.log_dir}/cov_mats"
        )
        order = "f-t"

        if self.dataset.env.config.num_rx_antennas > 1:
            order = "f-t-s"
        lmmse_int = LMMSEInterpolator(
            self.dataset.env.rg.pilot_pattern,
            time_cov_mat,
            freq_cov_mat,
            space_cov_mat,
            order=order,
        )
        lmmse_estimator = LSChannelEstimator(
            self.dataset.env.rg, interpolator=lmmse_int
        )

        almmse_h_hat, _ = lmmse_estimator((noisy_signals, noise_lin))

        return almmse_h_hat

    def evaluate_baselines(
        self,
        noisy_signals: tf.Tensor,
        noiseless_signals: tf.Tensor,
        channels: tf.Tensor,
        snr: int,
        baselines: List[str],
    ) -> Dict[str, float]:
        """Evaluates a list of baselines"""
        results = {}
        noise_lin = tf.pow(10.0, -snr / 10.0)

        for baseline in baselines:
            if baseline == "LS":
                h_ls_lin = self.evaluate_ls(noisy_signals)
                lin_mse = mse(tf.squeeze(channels), tf.squeeze(h_ls_lin))
                results["LS"] = lin_mse.numpy()
            elif baseline == "LMMSE":
                h_ls = self.evaluate_ls(noisy_signals, interpolate=False)
                h_lmmse = self.evaluate_lmmse(noiseless_signals, channels, h_ls, snr)
                lmmse_mse = mse(tf.squeeze(channels), tf.squeeze(h_lmmse))
                results["LMMSE"] = lmmse_mse.numpy()

            elif baseline == "ALMMSE":
                if isinstance(self.dataset.env, OfdmEnv):
                    pass
                almmse_h_hat = self.evaluate_almmse(noisy_signals, noise_lin)
                almmse_mse = mse(tf.squeeze(channels), tf.squeeze(almmse_h_hat))
                results["ALMMSE"] = almmse_mse.numpy()
            else:
                raise ValueError(f"Baseline is not supported {baseline}")

        return results

    def evaluate(
        self, snr_range: List[int], baselines: List[str] = [], save: bool = True
    ) -> None:
        """Evaluate a trained model"""

        test_loader = self.dataset.get_eval_loader(
            batch_size=self.config.eval_batch_size, setname="test"
        )
        apply_noiseless_channel = ApplyOFDMChannel(
            add_awgn=False, dtype=tf.as_dtype(tf.complex64)
        )
        apply_noisy_channel = ApplyOFDMChannel(
            add_awgn=True, dtype=tf.as_dtype(tf.complex64)
        )
        mses = pd.DataFrame(columns=["snr", "mse", "method", "seed"])

        if self.dataset.env is None:
            raise ValueError("Env cannot be None")

        for i, snr in enumerate(tqdm(snr_range)):
            noise_lin = tf.pow(10.0, -snr / 10.0)

            test_mse = defaultdict(list)
            num_steps = (
                len(self.dataset.test_indices[0]) + self.config.eval_batch_size - 1
            ) // self.config.eval_batch_size

            for step in range(num_steps):
                start, end = step * self.config.eval_batch_size, min(
                    (step + 1) * self.config.eval_batch_size,
                    len(self.dataset.test_indices[0]),
                )

                batch_size = min(self.config.eval_batch_size, end - start + 1)

                if self.dataset.x_samples is None:
                    symbols = self.dataset.env.generate_symbols(batch_size)
                else:
                    symbols = self.dataset.x_samples[i][self.dataset.test_indices[i]][
                        start:end
                    ]

                channels = self.dataset.test_y[i][start:end]

                if self.dataset.y_samples is None:
                    noisy_signals = apply_noisy_channel([symbols, channels, noise_lin])
                else:
                    noisy_signals = self.dataset.y_samples[i][
                        self.dataset.test_indices[i]
                    ][start:end]

                noiseless_signals = apply_noiseless_channel([symbols, channels])

                # evaluate the NN model
                inputs = self.dataset.env.estimate_at_pilot_locations(noisy_signals)
                pre_inputs = tf.map_fn(
                    self.dataset.preprocess_inputs,
                    inputs,
                    fn_output_signature=tf.float32,
                )
                h_hat = self.model(pre_inputs, training=False)
                h_hat_n = tf.map_fn(
                    postprocess, h_hat, fn_output_signature=tf.complex64
                )
                model_mse = mse(tf.squeeze(channels), h_hat_n).numpy()
                test_mse[self.config.model_name].append(model_mse)

                if len(baselines) > 0:
                    baseline_mses = self.evaluate_baselines(
                        noisy_signals, noiseless_signals, channels, snr, baselines
                    )

                    for baseline in baselines:
                        test_mse[baseline].append(baseline_mses[baseline])

            test_mse = {k: np.mean(v) for k, v in test_mse.items()}

            step = 1 + len(baselines)
            for mid, (k, v) in enumerate(test_mse.items()):

                mses.loc[step * i + mid] = [snr, v, k, self.config.seed]
        if save:
            mses.to_csv(os.path.join(self.log_dir, "test_mses.csv"), index=False)

        if self.config.verbose > 0:
            print(mses)

    def load_mdoel(self):
        self.model.load_weights(f"{self.log_dir}/cp.ckpt").expect_partial()
