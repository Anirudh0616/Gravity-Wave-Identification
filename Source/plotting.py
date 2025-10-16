import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
        ax[0, i].hist(mh_chain[:, i], bins=50, density=True)
        ax[0, i].set_title(f"{labels[i]}")
        if show_true:   ax[0, i].axvline(true_params[i], color='r', linestyle='--', label="True Value", linewidth= 3)
        ax[0, i].axvline(pred_value[i], color='g', linestyle='--', label='Median Value', linewidth= 3)
        ax[0, i].legend()
        ax[0, i].grid(True)
    ax[1, 0].plot(mh_chain[:, 0])
    ax[1, 1].plot(mh_chain[:, 1])
    ax[1, 2].plot(mh_chain[:, 2])
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


def corner_plot(true_params, mh_chain, labels, file: Path):
    n_dim = mh_chain.shape[1]
    show_true = True
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(4 * n_dim, 4 * n_dim))
    if np.isnan(true_params).any(): show_true = False
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if i == j:
                # Diagonal: parameter histogram
                ax.hist(mh_chain[:, i], bins=40, color="skyblue", alpha=0.7)
                ax.set_title(f"{labels[i]} Distribution", fontsize=15)
            elif i > j:
                # Lower triangle: scatter plot for parameter pairs
                ax.scatter(mh_chain[:, j], mh_chain[:, i], s=8, alpha=0.2, color="#008fd5")
                ax.grid(True, linestyle="--", linewidth=0.5)
                if show_true:
                    ax.scatter(true_params[j], true_params[i], marker='*', color='crimson', s=120, label="True Value")
                    ax.legend()
            else:
                # Upper triangle: turn off axis
                ax.set_axis_off()

            # Label axes
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=13)
            if i == n_dim - 1:
                ax.set_xlabel(labels[j], fontsize=13)

    plt.suptitle("Corner Plot of Parameters", fontsize=20)
    plt.tight_layout()
    path = Path(file)
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
