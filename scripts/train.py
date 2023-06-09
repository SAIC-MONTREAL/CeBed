"""
Train a deep channel estimator with CeBed
"""

import sys
import argparse
import numpy as np

from cebed.utils import set_random_seed
from cebed.trainer import Trainer, TrainConfig


def main(args):
    """Main function."""
    set_random_seed(args.seed)

    trainer = Trainer(TrainConfig(**vars(args)))
    trainer.setup()
    trainer.train()
    snr_range = np.arange(0, 25, 5)
    trainer.evaluate(snr_range=snr_range, baselines=["LS", "LMMSE", "ALMMSE"])
    trainer.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Channel Estimation")
    ########################## General args ###################################
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str)

    parser.add_argument("--dataset_name", type=str, default="SionnaOfflineMD")

    parser.add_argument("--model_name", type=str, default="ReEsNet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="train_output")
    ########################## Training params ################################

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--loss_fn", default="mse", type=str, help="The loss function")
    parser.add_argument("--train_split", type=float, default=0.9)

    parser.add_argument(
        "--early_stopping", action="store_true", help="If early stopping should be used"
    )

    parser.add_argument(
        "--input_type",
        type=str,
        help="The input type for data preprocessing",
        default=None,
        choices=["low", "raw"],
    )
    main(parser.parse_args(sys.argv[1:]))
