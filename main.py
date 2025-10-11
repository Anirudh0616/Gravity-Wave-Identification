import numpy as np
from Configurations.gw_functions import Gravitation_Wave
import Data_Gen.generator as generator
import Source.plotting as plot
from Source.metropolis_hasting import MetroHaste
from pathlib import Path


# Set True Values and Save Locations
true_params = [1.9, 9.5, 19.5]
data_path = Path("Data_Gen") / "Data_Grav_Wave.csv"
out_path_data = Path("Results/Plots/Gravitational_Wave_data.png")
out_path_pred = Path("Results/Plots/Gravitational_Wave_pred.png")
config_path = Path("Configurations") / "Grav_Wave.yaml"


# Generate Noisy Data
gw = Gravitation_Wave()
gw_timeseries = gw.Time_series(*true_params)
generator.Generate_Data(function= gw_timeseries, file_name = data_path, num = 500)

data = np.loadtxt(data_path, delimiter=",", skiprows= 1)

plot.data_points(data, gw_timeseries, out_path_data, "Input Noisy Data with True Parameter Model")

gw_parameter = gw.Parameter_Space(data[:, 0])
mh = MetroHaste(config_path, gw_parameter)

chain, diag = mh.MH_Solver(data)

print(f"acceptance rate: {diag["acceptance_rate"]}\n Parameter Predictions {diag['predicted_parameters']}")

plot.histogram_gw(true_params, chain, Path("Results/Plots/MH_hist.png"))

gw_pred_ts = gw.Time_series(*diag["predicted_parameters"])

plot.data_points(data, gw_pred_ts, out_path_pred, "Predicted Model with Noisy Datapoints")