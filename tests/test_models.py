import pytest
import yaml
import tensorflow as tf
import cebed
from cebed.models import ReEsNet, ChannelNet, MTRE, HA02


@pytest.mark.parametrize("input_shape", [[2, 72]])
@pytest.mark.parametrize(
    "experiment_name", ["siso_1_umi_block_1_ps2_p72", "simo_4_umi_block_1_ps2_p72"]
)
@pytest.mark.parametrize("model_name", ["ChannelNet", "ReEsNet", "InReEsNet", "HA02"])
def test_sr_model_outputs(input_shape, model_name, experiment_name):
    hparams_file = yaml.safe_load(open(f"./hyperparams/{model_name}.yaml", "r"))
    hparams = hparams_file[experiment_name]["default"]

    model_class = cebed.models.get_model_class(model_name)
    model = model_class(hparams)

    if experiment_name == "simo_4_umi_block_1_ps2_p72":
        n_channels = 4 * 2
    else:
        n_channels = 2
    inputs = tf.random.uniform(shape=[10] + input_shape + [n_channels])
    outputs = model(inputs)

    assert outputs.shape == [10] + hparams["output_dim"]


@pytest.mark.parametrize("input_shape", [[14, 72]])
@pytest.mark.parametrize(
    "experiment_name", ["siso_1_umi_block_1_ps2_p72", "simo_4_umi_block_1_ps2_p72"]
)
@pytest.mark.parametrize("model_name", ["DDAE", "MReEsNet", "MTRE"])
def test_masked_model_outputs(input_shape, model_name, experiment_name):
    hparams_file = yaml.safe_load(open(f"./hyperparams/{model_name}.yaml", "r"))
    hparams = hparams_file[experiment_name]["default"]

    model_class = cebed.models.get_model_class(model_name)
    model = model_class(hparams)

    if experiment_name == "simo_4_umi_block_1_ps2_p72":
        n_channels = 4 * 2
    else:
        n_channels = 2

    inputs = tf.random.uniform(shape=[10] + input_shape + [n_channels])
    outputs = model(inputs)

    assert outputs.shape == [10] + hparams["output_dim"]
