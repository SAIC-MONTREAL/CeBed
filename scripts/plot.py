import sys
import os
from typing import List
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import glob

from cebed.utils import read_metadata

sns.set_style("whitegrid")
sns.set_context("paper")


def compute_normalized_score(df: pd.DataFrame, model_name: str) -> float:
    """
    Computes normalized score as defined in Eq (1)
    :param df: Dataframe containing the MSE all the models and the baselines
    :param model_name: Model name

    :return The normalized score
    """
    lmmse_mse = df[df.method == "LMMSE"].mse.values
    ls_mse = df[df.method == "LS"].mse.values
    nn_mse = df[df.method == model_name].mse.values
    scores = np.maximum((nn_mse - ls_mse) / (lmmse_mse - ls_mse), 0.0)
    return scores * 100


def get_filenames(paths: List[str]) -> List[str]:
    """Gets all the result files"""
    filenames = []
    for path in paths:
        filenames.extend(glob.glob(f"{path}/*/*/*/test_mse*.csv"))

    return filenames


def read_results(filenames):
    results = pd.DataFrame()

    for filename in filenames:
        parent = Path(filename).parents[0]

        df = pd.read_csv(filename)
        config = read_metadata(os.path.join(parent, "config.yaml"))

        df["task"] = config.get("experiment_name", "Exp")
        model_name = filename.split("/")[-2]

        if "p_spacing" in config:
            num_pilots = int(config["fft_size"] / config["p_spacing"]) * len(
                config["pilot_ofdm_symbol_indices"]
            )
            df["num_pilots"] = num_pilots

        df["nr"] = config.get("num_rx_antennas", None)
        df["speed"] = config.get("ue_speed", None)
        scores = compute_normalized_score(df, model_name=model_name)
        df.loc[df.method == model_name, "score"] = scores

        results = pd.concat([results, df], ignore_index=True)
    return results


def plot():
    os.makedirs("./benchmark/figures", exist_ok=True)
    paths = sys.argv[1:-1]
    title = sys.argv[-1]

    filenames = get_filenames(paths)
    results = read_results(filenames)

    methods = sorted(results.method.unique())
    order = sorted(results.method.unique())

    mks = ["o", "<", "^", ">", "*", "8", "s", "p", "d", "h"]

    palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(methods))))
    markers = dict(zip(order, mks))

    x_values = results.snr.unique()

    # MSE Figure
    figure = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    sns.lineplot(
        x="snr",
        y="mse",
        hue="method",
        style="method",
        markers=markers,
        dashes=False,
        data=results,
        hue_order=order,
        palette=palette,
        ax=ax,
    )
    ax.set(yscale="log")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    ax.set_xticks(x_values, list(map(int, x_values)))
    ax.set_xlim(int(min(x_values)), int(max(x_values)))
    ax.set_ylim(0.0001, max(results.mse))
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=5, frameon=False)
    plt.tight_layout()
    plt.savefig(f"./benchmark/figures/mse_{title}.png")

    # Score Figure
    plt.cla()
    y = "score"
    figure = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    score_data = results[~results.method.isin(["LMMSE", "LS", "ALMMSE"])]
    norder = (
        score_data.groupby("method", as_index=False)
        .mse.mean()
        .sort_values("mse", ascending=False)
        .method
    )

    sns.lineplot(
        x="snr",
        y=y,
        hue="method",
        style="method",
        dashes=False,
        markers=markers,
        hue_order=norder,
        data=score_data,
        palette=palette,
        ax=ax,
        legend=False,
    )
    ax.set_ylabel("Normalized Score")
    ax.set_xlabel("SNR (dB)")
    ax.set_xticks(x_values, list(map(int, x_values)))
    ax.set_xlim(int(min(x_values)), int(max(x_values)))
    ax.set_ylim(0, 100)
    ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.2)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=7, frameon=False)
    figure.tight_layout()
    plt.savefig(f"./benchmark/figures/score_{title}.png")

    # # Figure Bar plot

    mean_mse = (
        results.groupby("method", as_index=False)
        .mse.mean()
        .sort_values("mse", ascending=False)
    )
    mse_ls = mean_mse[mean_mse.method == "LS"].mse.values
    vals = mean_mse[~mean_mse.method.isin(["LS"])].mse.values
    names = mean_mse.method.values
    gains = 10 * np.log10(mse_ls / vals)
    figure = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)

    order = (
        results.groupby("method", as_index=False)
        .mse.mean()
        .sort_values("mse", ascending=False)
        .method
    )
    sns.barplot(x="method", y="mse", data=results, order=order, palette=palette, ax=ax)
    id = 0
    for i, patch in enumerate(ax.patches):
        if names[i] in ["LS"]:
            continue
        h, w, x = patch.get_height(), patch.get_width(), patch.get_x()
        xy = (x + w / 2.0, h + h / 2)
        text = f"-{gains[id]:0.1f} dB"
        ax.annotate(text=text, xy=xy, ha="center", va="bottom", weight="bold")
        id += 1

    ax.set(yscale="log")
    ax.set(xlabel=None)
    ax.set_ylabel("MSE")
    ax.grid(True, which="both", linestyle="--", color="gray")
    figure.tight_layout()
    plt.savefig(f"./benchmark/figures/bar_{title}.png")


if __name__ == "__main__":
    plot()
