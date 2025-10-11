import numpy as np
from Configurations.gw_functions import Gravitation_Wave
import Data_Gen.generator as generator
import Source.plotting as plot
from Source.metropolis_hasting import MetroHaste
from pathlib import Path

data_path = Path("Data_Gen") / "Data_Grav_Wave.csv"
data = np.loadtxt(data_path, delimiter=",", skiprows= 1)
true_params = [0.3, 5, 10]
gw = Gravitation_Wave()
gw_timeseries = gw.Time_series(*true_params)

out_path_data = Path("Results/Plots/Gravitational_Wave_data.png")
plot.data_points(data, gw_timeseries, out_path_data)

config_path = Path("Configurations") / "Grav_Wave.yaml"
gw_parameter = gw.Parameter_Space(data[:, 0])
mh = MetroHaste(config_path, gw_parameter)

chain, diag = mh.MH_Solver(data)

print(diag)

plot.histogram_gw(true_params, chain, Path("Results/Plots/MH_hist.png"))

gw_pred_ts = gw.Time_series(*diag["predicted_parameters"])
out_path_pred = Path("Results/Plots/Gravitational_Wave_pred.png")
plot.data_points(data, gw_pred_ts, out_path_pred)