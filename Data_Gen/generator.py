import numpy as np
import pandas as pd
from pathlib import Path
EPS = 1e-9 # Protection division by zero

# Create Noisy Data for given set of Parameters

def Generate_Data(function, file_name: Path, noise = 0.2, t_min: float = 0, t_max: float = 12, num = 1000):
    """
    Generates Time Series Noisy Data for given function

    :param function: Input Model Function
    :param file_name: Name of file to Save
    :param noise: Noise in Datapoints
    :param t_min: time lower bound
    :param t_max: time upper bound
    :param num: Number of Datapoints
    :return: Time Series Data ( Time, Datapoints )
    """

    file_path = Path(file_name)
    if file_path.suffix != ".csv":
        file_path = file_path.with_suffix(".csv")

    time = np.linspace(t_min, t_max, num)
    f_points = function(time)
    f_error = noise * f_points + 0.1 * np.std(f_points)
    f_noisy = f_points + np.random.normal(0, np.abs(f_error), num)

    df = pd.DataFrame({
        'time': time,
        'datapoint': f_noisy
    })
    # ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # overwrite file safely
    df.to_csv(file_path, index=False, float_format="%.6f")

    return time, f_noisy

# Generate_Data(gwf.Create_TimeMod_GW(0.3, 5, 10, 7.5),"Grav_Wave_TimeModded",
#               noise = 0.2, t_min = -7.5, t_max = 7.5, num = 1001)
