from cebed.models.transformers import HA02, MTRE
from cebed.models.autoencoders import DDAE

from cebed.models.residual import ReEsNet, InReEsNet, MReEsNet
from cebed.models.cnn import ChannelNet
import yaml
from pathlib import Path

MODELS = ["MReEsNet", "ChannelNet", "ReEsNet", "InReEsNet", "HA02", "MTRE", "DDAE"]


def get_model_class(model_name):

    if model_name not in globals() or model_name not in MODELS:
        raise NotImplementedError("Model not found: {}".format(model_name))

    return globals()[model_name]


def get_model_hparams(model_name, experiment_name):
    """
    Returns the model hyperparamters
    :param model_name: The name of the model. Musy be in MODELS
    :param experiment_name: The experiment name
    """

    with open(f"./hyperparams/{model_name}.yaml") as f:
        model_hparams = yaml.safe_load(f)

    if "best" in model_hparams[experiment_name]:
        model_hparams = model_hparams[experiment_name]["best"]
    else:
        model_hparams = model_hparams[experiment_name]["default"]

    return model_hparams
