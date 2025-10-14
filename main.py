import numpy as np
from pathlib import Path
import argparse

from Configurations.gw_functions import Gravitation_Wave
import Data_Gen.generator as generator
import Source.plotting as plot
from Source.metropolis_hasting import MetroHaste


# Labels for params and Save Locations
labels = ['alpha', 'beta', 'gamma']
data_path = Path("Data_Gen") / "Data_Grav_Wave.csv"
out_path = Path("Results")
path_data_plots = Path("Gravitational_Wave_data.png")
path_pred_plots = Path("Gravitational_Wave_pred.png")
config_path = Path("Configurations") / "Grav_Wave.yaml"

def run_generated_data(alpha: float, beta: float, gamma: float, name: str):
    # Display True Values
    true_params = [alpha, beta, gamma]
    # Generate Noisy Data
    gw = Gravitation_Wave()
    gw_timeseries = gw.Time_series(*true_params)
    generator.Generate_Data(function= gw_timeseries, file_name = data_path, num = 500)

    data = np.loadtxt(data_path, delimiter=",", skiprows= 1)
    out_path_data = out_path / name / path_data_plots
    plot.data_points(data, gw_timeseries, out_path_data, "Input Noisy Data with True Parameter Model")

    gw_parameter = gw.Parameter_Space(data[:, 0])
    mh = MetroHaste(config_path, gw_parameter)

    chain, diag = mh.MH_Solver(data)
    print("\n")
    print("--" * 10)
    print("True Params ", end="")
    for i in range(3): print(f"{labels[i]}: {true_params[i]}", end="  ")
    print("\n")
    print(f"acceptance rate: {diag["acceptance_rate"]}")
    q_lo, q_hi = np.quantile(chain, [0.025, 0.975], axis=0)
    median = diag["pred_params"]
    for lab, m, lo, hi in zip(labels, median, q_lo, q_hi):
        print(f"{lab}:\n \tmedian={m:.3f}\n \t95% Credibility interval=( {lo:.3f}, {hi:.3f} )")

    plot.histogram_gw(true_params, chain, out_path / name / Path("MH_hist.png") )

    gw_pred_ts = gw.Time_series(*diag["pred_params"])
    out_path_pred = out_path / name / path_pred_plots
    plot.data_points(data, gw_pred_ts, out_path_pred, "Predicted Model with Noisy Datapoints")

def parse_args():
    p = argparse.ArgumentParser(description="Run GW MCMC experiment")
    p.add_argument("--alpha", type=float, required=True, help="alpha in (0,2)")
    p.add_argument("--beta", type=float, required=True, help="beta in (1,10)")
    p.add_argument("--gamma", type=float, required=True, help="gamma in (1,20)")
    p.add_argument("--id", type=str, required=True, help="ID of experiment")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not (0.0 < args.alpha < 2.0):
        raise SystemExit("alpha must be in (0,2)")
    if not (1.0 < args.beta < 10.0):
        raise SystemExit("beta must be in (1,10)")
    if not (1.0 < args.gamma < 20.0):
        raise SystemExit("gamma must be in (1,20)")

    run_generated_data(args.alpha, args.beta, args.gamma, args.id)