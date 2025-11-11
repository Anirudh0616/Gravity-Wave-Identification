import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import yaml

from Configurations.gw_functions import Gravitation_Wave
import Data_Gen.generator as generator
import Source.plotting as plot
from Source.metropolis_hasting import MetroHaste


# Labels for params and Save Locations
labels = ['Alpha', 'Beta', 'Gamma']
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
    generator.Generate_Data(function= gw_timeseries, file_name = data_path, num = 1500)

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
    for l, e in zip(labels, diag["ESS"]):
        print(f"ESS({l}) = {e:.1f}")   
    print(f"acceptance rate: {diag["acceptance_rate"]}")
    q_lo, q_hi = np.quantile(chain, [0.025, 0.975], axis=0)
    median = diag["pred_params"]
    for lab, m, lo, hi in zip(labels, median, q_lo, q_hi):
        print(f"{lab}:\n \tmedian={m:.3f}\n \t95% Credibility interval=( {lo:.3f}, {hi:.3f} )")

    plot.histogram_gw(true_params, chain, out_path / name / Path("MH_hist.png") )
    plot.corner_plot(true_params, chain, labels , out_path / name / Path("MH_corner.png"))

    gw_pred_ts = gw.Time_series(*diag["pred_params"])
    out_path_pred = out_path / name / path_pred_plots
    plot.data_points(data, gw_pred_ts, out_path_pred, "Predicted Model with Noisy Datapoints")


def run_unknown(unknown_path: Path = Path("gw_data.csv")):
    # Load and Clean Data
    df = pd.read_csv(unknown_path)
    df = df.drop('Unnamed: 1', axis=1)
    t_array = df['t'].to_numpy()
    h_array = df['h'].to_numpy()
    data = np.column_stack([t_array, h_array])

    # Start Metropolis Hastings for Gravitational Wave
    gw = Gravitation_Wave()
    gw_parameter = gw.Parameter_Space(data[:, 0])
    mh = MetroHaste(config_path, gw_parameter)

    chain, diag = mh.MH_Solver(data)

    # Predicted Wave Time Series Data
    gw_pred_ts = gw.Time_series(*diag["pred_params"])

    # Plotting
    true_params = np.array([np.nan, np.nan, np.nan]) # Unknown True Parameters
    name = Path("Original_Unknown") # Experiment Name of Unknown Data
    plot.histogram_gw(true_params, chain, out_path / name / Path("Histogram.png"))
    plot.corner_plot(true_params, chain, labels, out_path / name / Path("Covariance"))

    out_path_pred = out_path / name / path_pred_plots
    plot.data_points(data, gw_pred_ts, out_path_pred, "Predicted Model with Noisy Datapoints")

    # Calculating Noise
    prediction_ts = gw_pred_ts(t_array)
    residual = h_array - prediction_ts

    # Global Signal to Noise Ratio
    signal_rms = np.sqrt(np.mean((prediction_ts)**2))
    noise_rms = np.sqrt(np.mean((residual)**2))
    snr_global = signal_rms / noise_rms
    
    center_mask = (t_array > 2.7) & (t_array < 4.2)
    # print(t_array[center_mask])
    center_signal_rms = np.sqrt(np.mean((prediction_ts[center_mask])**2))
    center_noise_rms = np.sqrt(np.mean((residual[center_mask])**2))
    snr_local = center_signal_rms / center_noise_rms

    # printing
    for l, e, m in zip(labels, diag["ESS"], diag["MCSE"]):
        print(f"ess({l}) = {e:.1f}")   
        print(f"MCSE({l}) = {m:.5f}")   
        
    print(f"acceptance rate: {diag["acceptance_rate"]}")
    print(f"Global Signal to Noise Ratio: {snr_global:.2f}")
    print(f"Local Signal to Nosie Ratio: {snr_local:.2f}")
    q_lo, q_hi = np.quantile(chain, [0.025, 0.975], axis=0)
    median = diag["pred_params"]
    for lab, m, lo, hi in zip(labels, median, q_lo, q_hi):
        print(f"{lab}:\n \tmedian={m:.3f}\n \t95% credibility interval=( {lo:.3f}, {hi:.3f} )")


def variance_test(alpha: float, beta: float, gamma: float):
    config_variance = Path("configurations/variance_test.yaml")
    with open(config_variance, "r") as f:
        cfg = yaml.safe_load(f)

    base = np.array(cfg["proposal_scales"])
    multipliers = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 50.0, 100.0])

    results = []
    
    for m in multipliers:
        proposal_vec = base * m
        print(f"running multiplier {m:.2f}")
        true_params = [alpha, beta, gamma]
        # generate noisy data
        gw = Gravitation_Wave()
        gw_timeseries = gw.Time_series(*true_params)
        generator.Generate_Data(function= gw_timeseries, file_name = data_path, num = 1500)

        data = np.loadtxt(data_path, delimiter=",", skiprows= 1)

        gw_parameter = gw.Parameter_Space(data[:, 0])
        mh = MetroHaste(config_variance, gw_parameter)
        mh.scales = proposal_vec

        chain, diag = mh.MH_Solver(data)
        acceptance_rate = diag["acceptance_rate"]
        prediction = diag["pred_params"]
        accuracy = 1- np.linalg.norm(prediction - true_params) / np.linalg.norm(true_params)

        results.append({
            "multiplier": m,
            "scales": proposal_vec.tolist(),
            "acceptance": acceptance_rate,
            "accuracy": accuracy,
            })
    
    plot.variance_plot(results)

    return results





def parse_args():
    parser = argparse.ArgumentParser(description="run gravitational wave analysis")
    parser.add_argument('--mode', choices=['generated', 'unknown','variance'], required=True,
                        help='mode to run: generated or unknown')
    parser.add_argument('--alpha', type=float, default=None, help='alpha parameter (required for generated)')
    parser.add_argument('--beta', type=float, default=None, help='beta parameter (required for generated)')
    parser.add_argument('--gamma', type=float, default=None, help='gamma parameter (required for generated)')
    parser.add_argument('--id', type=str, default=None, help='experiment id (optional)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'generated':
        if args.alpha is None or args.beta is None or args.gamma is None:
            raise SystemExit("for generated mode, alpha, beta, and gamma are required")
        if not (0.0 < args.alpha < 2.0):
            raise SystemExit("alpha must be in (0,2)")
        if not (1.0 < args.beta < 10.0):
            raise SystemExit("beta must be in (1,10)")
        if not (1.0 < args.gamma < 20.0):
            raise SystemExit("gamma must be in (1,20)")
        run_generated_data(args.alpha, args.beta, args.gamma, args.id)

    elif args.mode == 'unknown':
        unknown_file = Path("gw_data.csv")
        run_unknown(unknown_file)

    elif args.mode == 'variance':
        if args.alpha is None or args.beta is None or args.gamma is None:
            raise SystemExit("for generated mode, alpha, beta, and gamma are required")
        if not (0.0 < args.alpha < 2.0):
            raise SystemExit("alpha must be in (0,2)")
        if not (1.0 < args.beta < 10.0):
            raise SystemExit("beta must be in (1,10)")
        if not (1.0 < args.gamma < 20.0):
            raise SystemExit("gamma must be in (1,20)")
        result = variance_test(args.alpha, args.beta, args.gamma)
#        print(result)

    else:
        raise SystemExit("invalid mode; must be 'generated' or 'unknown' or 'variance' ")
