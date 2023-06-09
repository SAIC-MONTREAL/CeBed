"""
Evaluate a model trained with CeBed
"""

import sys
import os
import argparse
import numpy as np

from cebed.trainer import Trainer, TrainConfig
from cebed.utils import read_metadata, set_random_seed

if __name__ == "__main__":
    model_path = sys.argv[1]
    baselines = sys.argv[2:]

    saved_config = read_metadata(os.path.join(model_path, "config.yaml"))
    config = TrainConfig()

    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)
    print(config)

    set_random_seed(config.seed)
    trainer = Trainer(config)
    trainer.setup()
    trainer.load_mdoel()
    snr_range = np.arange(0, 25, 5)
    trainer.evaluate(snr_range=snr_range, baselines=baselines, save=False)
