import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict

from seaborn import JointGrid

def histogram_gw(true_params, mh_chain, file: Path):
    fig, ax = plt.subplots(2, 3, figsize=(24, 12))
    show_true = True
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.isnan(true_params).any(): show_true = False
    labels = ['alpha', 'beta', 'gamma']
    # Predicted Value
    pred_value = np.median(mh_chain, axis=0)
    for i in range(3):
        # Compute histogram first
        counts, bins = np.histogram(mh_chain[:, i], bins=50)

        # Plot the histogram
        ax[0, i].hist(mh_chain[:, i], bins=50, density=True, color = "darkorange")
        ax[0, i].set_title(f"{labels[i]}")
        if show_true:   ax[0, i].axvline(true_params[i], color='r', linestyle='--', label="True Value", linewidth= 3)
        ax[0, i].axvline(pred_value[i], color='g', linestyle='--', label='Median Value', linewidth= 3)
        ax[0, i].legend()
        ax[0, i].grid(True)
    ax[1, 0].plot(mh_chain[:, 0], color = "darkorange")
    ax[1, 1].plot(mh_chain[:, 1], color = "darkorange")
    ax[1, 2].plot(mh_chain[:, 2], color = "darkorange")
    plt.tight_layout()

    path = Path(file)
    plt.savefig(path, bbox_inches='tight')

def data_points(datapoints, function, file: Path, title: str):
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.scatter(datapoints[:, 0], datapoints[:, 1], label="Datapoints", s = 10, c='r', marker='o')
    model_data = function(datapoints[:, 0])
    ax.plot(datapoints[:, 0], model_data, label="Model function", color='b', linestyle='--')
    ax.grid(True, linestyle='-', linewidth='0.5')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude of Gravitational Wave")
    ax.legend()

    # Ensure a proper extension; .png is a good default
    if path.suffix == "":
        path = path.with_suffix(".png")
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight", dpi=200)

    plt.close(fig)


def corner_plot(true_params, mh_chain, labels, out_dir: Path):
    """
    Creates pairwise joint plots for all parameter combinations.
    Each plot is saved individually as a PNG in the given output directory.
    """
    def plot_joinplot(g: JointGrid, i: int, j: int):
        g.set_axis_labels(labels[i], labels[j])

        g.ax_joint.scatter(pred_value[i], pred_value[j], color="blue", marker="+", s=100, label="Predicted")
        if show_true:
            g.ax_joint.scatter(true_params[i], true_params[j], color="limegreen", s=60, marker="x", label="True")

        g.ax_joint.legend()
        g.figure.suptitle(f"Covariance: {labels[i]} vs {labels[j]}", fontsize=14)
        g.figure.tight_layout()
        out_path = out_dir / f"{labels[i]}_vs_{labels[j]}.png"
        g.figure.savefig(out_path, bbox_inches="tight")
        plt.close(g.figure)

    n_dim = mh_chain.shape[1]
    show_true = not np.isnan(true_params).any()
    pred_value = np.median(mh_chain, axis=0)
    range_alpha = np.quantile(mh_chain[:, 0], [0.0005, 0.9995])
    range_beta = np.quantile(mh_chain[:, 1], [0.00001, 0.99999])
    range_gamma = np.quantile(mh_chain[:, 2], [0.00001, 0.99999])

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", color_codes=True)
    i = 0 # alpha
    j = 1 # beta
    g = sns.jointplot(
        x=mh_chain[:, i],
        y=mh_chain[:, j],
        kind="hex",
        cmap="Oranges",
        color="darkorange",
        space=0,
        marginal_ticks=True,
        bins = "log",
        alpha = 0.8,
        gridsize = 40,
        xlim = range_alpha,
        ylim = range_beta
    )
    plot_joinplot(g, i, j)

    i = 1  # beta
    j = 2  # gamma
    g = sns.jointplot(
        x=mh_chain[:, i],
        y=mh_chain[:, j],
        kind="hex",
        cmap="Oranges",
        color="darkorange",
        space=0,
        marginal_ticks=True,
        bins = "log",
        alpha=0.8,
        gridsize=40,
        xlim=range_beta,
        ylim=range_gamma
    )
    plot_joinplot(g, i, j)

    i = 2  # gamma
    j = 0  # alpha
    g = sns.jointplot(
        x=mh_chain[:, i],
        y=mh_chain[:, j],
        kind="hex",
        cmap="Oranges",
        color="darkorange",
        space=0,
        marginal_ticks=True,
        bins = "log",
        alpha=0.8,
        gridsize=40,
        xlim=range_gamma,
        ylim=range_alpha
    )
    plot_joinplot(g, i, j)


def variance_plot(results: List[Dict]):
    path = Path("Results/Variance_Test.png")
    scales = [r["multiplier"] for r in results]
    acc_rates = [r["acceptance"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, ax1 = plt.subplots()
    ax1.set_xscale("log")
    ax1.set_xlabel("Scale Multiplier")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(scales, accuracies, color="tab:blue", marker="o", label="Accuracy")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Acceptance Rate", color="tab:red")
    ax2.plot(scales, acc_rates, color="tab:red", marker="s", label="Acceptance")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Metropolis–Hastings: Accuracy & Acceptance vs Scale Multiplier")

    fig.savefig(path, bbox_inches="tight", dpi=200)

