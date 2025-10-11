import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def histogram_gw(true_params, mh_chain, file: Path):
    fig, ax = plt.subplots(2, 3, figsize=(24, 12))
    labels = ['alpha', 'beta', 'gamma']

    for i in range(3):
        # Compute histogram first
        counts, bins = np.histogram(mh_chain[:, i], bins=50)

        # Plot the histogram
        ax[0, i].hist(mh_chain[:, i], bins=50, density=True)
        ax[0, i].set_title(f"{labels[i]}")
        ax[0, i].axvline(true_params[i], color='r', linestyle='--', label="True Value")
        # Pred value
        max_count_idx = np.argmax(counts)
        pred_value = (bins[max_count_idx] + bins[max_count_idx + 1]) / 2

        ax[0, i].axvline(pred_value, color='g', linestyle='--', label='Predicted Value')
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