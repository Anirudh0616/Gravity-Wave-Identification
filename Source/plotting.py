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
        ax[0, i].set_title(f"{labels[i]}, seed = {seed}")
        ax[0, i].axvline(true_params[i], color='r', linestyle='--', label="True Value")
        # Pred value
        max_count_idx = np.argmax(counts)
        pred_value = (bins[max_count_idx] + bins[max_count_idx + 1]) / 2

        ax[0, i].axvline(pred_value, color='g', linestyle='--', label='Predicted Value')
        print(f"Predicted Value of {labels[i]} at {true_params[i]} is {pred_value:.2f}")
        ax[0, i].legend()
        ax[0, i].grid(True)
    ax[1, 0].plot(mh_chain[:, 0])
    ax[1, 1].plot(mh_chain[:, 1])
    ax[1, 2].plot(mh_chain[:, 2])
    plt.tight_layout()

    path = Path(file)
    plt.savefig(path, bbox_inches='tight')

def data_points(datapoints, function, file: Path):
    plt.figure(figsize=(24, 12))
    plt.scatter(datapoints[:, 0], datapoints[:, 1], label = "Datapoints")
    model_data = function(datapoints[:, 0])
    plt.plot(datapoints[:, 0], model_data, label = "Model function")
    path = Path(file)
    plt.savefig(path, bbox_inches='tight')