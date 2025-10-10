import matplotlib.pyplot as plt

def GW_Plot():
    fig, ax = plt.subplots(2, 3, figsize=(24, 12))
    labels = ['alpha', 'beta', 'gamma']
    for j in range(1):
        seed = int(1000*np.random.random())
        print(seed)
        MH_chain, Proposal_chain = MHSampling(seed=seed)
        for i in range(3):
            # Compute histogram first
            counts, bins = np.histogram(MH_chain[:, i], bins=50)

            # Plot the histogram
            ax[0, i].hist(MH_chain[:, i], bins=50, density=True)
            ax[0, i].set_title(f"{labels[i]}, seed = {seed}")
            ax[0, i].axvline(true_params[i], color='r', linestyle='--', label="True Value")

            # Find the bin with maximum count (mode)
            max_count_idx = np.argmax(counts)
            # Get the center of that bin
            pred_value = (bins[max_count_idx] + bins[max_count_idx + 1]) / 2

            ax[0, i].axvline(pred_value, color='g', linestyle='--', label='Predicted Value')
            print(f"Predicted Value of {labels[i]} at {true_params[i]} is {pred_value:.2f}")
            ax[0, i].legend()
            ax[0, i].grid(True)
        rejection_plots = False
        ax[1, 0].plot(MH_chain[:, 0])
        if rejection_plots: ax[1, 0].scatter(np.arange(0,MH_chain.shape[0], 1), Proposal_chain[:, 0], s = 0.2)
        ax[1, 1].plot(MH_chain[:, 1])
        if rejection_plots: ax[1, 1].scatter(np.arange(0,MH_chain.shape[0], 1),Proposal_chain[:, 1], s = 0.2)
        ax[1, 2].plot(MH_chain[:, 2])
        if rejection_plots: ax[1, 2].scatter(np.arange(0,MH_chain.shape[0], 1), Proposal_chain[:, 2], s = 0.2)
    plt.tight_layout()
    plt.show()