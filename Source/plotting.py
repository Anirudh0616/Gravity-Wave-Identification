import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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

    # fig, axes = plt.subplots(nrows=1, ncols=n_dim, figsize=(4 * n_dim, 4 * n_dim))
    # for i in range(n_dim):
    #     for j in range(n_dim):
    #         ax = axes[i, j]
    #         if i == j:
    #             # Diagonal: parameter histogram
    #             ax.hist(mh_chain[:, i], bins=40, color="skyblue", alpha=0.7)
    #             ax.set_title(f"{labels[i]} Distribution", fontsize=15)
    #         elif i > j:
    #             # Lower triangle: scatter plot for parameter pairs
    #             ax.scatter(mh_chain[:, j], mh_chain[:, i], s=8, alpha=0.2, color="#008fd5")
    #             ax.grid(True, linestyle="--", linewidth=0.5)
    #             ax.scatter(pred_value[j], pred_value[i], marker='o', color='green', label="Predicted Value")
    #             if show_true:
    #                 ax.scatter(true_params[j], true_params[i], marker='*', color='crimson', s=120, label="True Value")
    #             ax.legend()
    #         else:
    #             # Upper triangle: turn off axis
    #             ax.set_axis_off()
    #
    #         # Label axes
    #         if j == 0 and i != 0:
    #             ax.set_ylabel(labels[i], fontsize=13)
    #         if i == n_dim - 1:
    #             ax.set_xlabel(labels[j], fontsize=13)


    # plt.suptitle("Covariance Plot of Parameters", fontsize=20)
    # plt.tight_layout()
    # path = Path(file)
    # path.parent.mkdir(exist_ok=True, parents=True)
    # plt.savefig(path, bbox_inches="tight", dpi=200)
    # plt.close(fig)

if __name__ == "__main__":
    np.random.seed(42)

    # Simulate a fake MCMC chain with 3 parameters
    n_samples = 1000
    n_params = 3
    mh_chain = np.random.randn(n_samples, n_params) * [1.0, 2.0, 0.5] + [0.5, -1.0, 2.0]

    # True parameter values (for reference)
    true_params = np.array([0.5, -1.0, 2.0])

    # Labels for each parameter
    labels = ["Alpha", "Beta", "Gamma"]

    # Output file
    output_file = Path("test_corner_plot")

    # Call the plotting function
    corner_plot(true_params, mh_chain, labels, output_file)
