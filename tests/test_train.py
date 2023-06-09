"""
Tests to make sure training runs
"""

import pytest
from cebed.trainer import Trainer, TrainConfig


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_train():
    """test train works properly"""
    config = TrainConfig()
    config.epochs = 1
    config.data_dir = "tests/data/siso_data_md"
    config.experiment_name = "siso_1_umi_block_1_ps2_p72"

    trainer = Trainer(config)

    trainer.setup()

    trainer.train()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_evaluate():
    """test evaluate works properly"""
    config = TrainConfig()
    config.epochs = 1
    config.data_dir = "tests/data/siso_data_md"
    config.experiment_name = "siso_1_umi_block_1_ps2_p72"
    trainer = Trainer(config)

    trainer.setup()

    trainer.evaluate(snr_range=[0], baselines=["LS", "LMMSE"])
