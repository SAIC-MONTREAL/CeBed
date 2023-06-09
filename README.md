<div align='center'>
<p align='center'>
  <img width='50%' src='./.assets/banner.png' />
</p>


![Continuous Integration](https://github.com/SAIC-MONTREAL/CeBed/actions/workflows/python-package.yml/badge.svg)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

Channel estimation test bed (CeBed) is a suite of implementations and benchmarks for OFDM channel estimation in Tensorflow.

The goal of CeBed is to unify and facilitate the replication, refinement and design of new deep channel estimators. CeBed can also be used as a baseline to create and build new projects and compare with existing algorithms in the litterature. It is simple to add a new dataset or a model to our package and we welcome the community to update or add exisiting algorithms or datasets.

For now, CeBed provides a simple interface to train and evaluate various deep channel estimation models.

# Setup

<details open>

Clone repo and install the requirements in a [**Python>=3.8.0**](https://www.python.org/) environment.
```bash
git clone https://github.com/SAIC-MONTREAL/CeBed
cd CeBed
pip install -e .
```
</details>

<!--# Setup
CeBed can be installed as follows
-->

# Using CeBed
## Datasets
### Sionna dataset
<details>
For now, CeBed uses the link-level simulator [Sionna](https://nvlabs.github.io/sionna/) for data generation. CeBed provides an interface to generate datasets using different channel models, system parameters, pilot patterns, etc.

Here is an example to generate a `SISO` dataset using one SNR level (by default = 0 dB) :
```bash
python scripts/generate_datasets_from_sionna.py --size 10000 --num_rx_antennas 1 --path_loss
```
The generated dataset contains:
- `x`: Transmitted symbols, a complex tensor with shape `[batch_size, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]`
- `h`: The channel impulse response, a complex tensor with shape `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]`
- `y`: The received symbols, a complex tensor with shape `[batch size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]`

Here is another example on how to generate a multi-domain dataset where each SNR level is a different domain:
```bash
python scripts/generate_datasets_from_sionna.py --size 10000 --scenario umi --num_rx_antennas 1 --path_loss --num_domains 5 start_ds 0 end_ds 25
```
</details>

### Custom dataset
<details>
It is easy to add a new dataset to CeBed. The dataset can be generated offline using any link-level simulator like MATLAB.

Please check the tutorial in [notebooks/cusotm_dataset.ipynb](notebooks/custom_dataset.ipynb), detailing how to use CeBed with your dataset.
</details>

## Training

<details>
<summary>Single model training</summary>

The command below will train and evaluate a single model
```bash
python scripts/train.py --experiment_name EXPERIMENT_NAME --seed SEED --data_dir DATADIR --epochs 100 --dataset_name SionnaOfflineMD --model_name ReEsNet --input_type low
```
</details>

<details>
<summary>Model hyperparameters</summary>

The model hyperprameters are defined in `yaml` files under [hyperparams](./hyperparams).
Make sure that the `EXPERIMENT_NAME` exists in the yaml files of the model(s) you would like to train.
Here is an example configuration of the [ReEsNet model](./hyperparams/ReEsNet.yaml):
```yaml
MyExperimentName:
  default:
    hidden_size: 16
    input_type: low
    kernel_size: 3
    lr: 0.001
    n_blocks: 4
    upsamling_mode: deconv
```
</details>

<details>
<summary>Benchmarking all models</summary>

To reproduce the benchamrking results from our paper:
```python
python scripts/benchmark.py --seed SEED --data_dir DATADIR --epochs 100 --experiment_name EXPERIMENT_NAME --gpus GPU_IDS
```

**Note**: The model inputs and outputs are expects to have the following shape `[batch_size, num_ofdm_symbols, num_ofdm_subcarriers, num_channels]` where `num_channels = num_rx_ant*num_tx*2`.

</details>

## Evaluation
<details>
**Evaluate a trained model**

To evaluate a model trained with CeBed,
```
python scripts/evaluate.py PATH_TO_MODEL
```

**Evaluate model and baselines**

You can provide a list of baselines to compare the model to :
```
python scripts/evaluate.py PATH_TO_MODEL LS LMMSE ALMMSE
```
</details>


# Citation
If you use our code, please cite our work.
```bibtex
@article{cebed,
  author  = {Amal Feriani and Di Wu and Steve Liu and Greg Dudek},
  title   = {CeBed: A Benchmark for Deep Data-Driven OFDM Channel Estimation},
  url     = {https://github.com/SAIC-MONTREAL/cebed.git}
  yeart   = {2023}
}
```

# License

The code is licensed under the [Creative Commons Attribution 4.0 License (CC BY)](https://creativecommons.org/licenses/by/4.0/).
