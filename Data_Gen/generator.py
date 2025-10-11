import numpy as np
import pandas as pd
import Configurations.gw_functions as gwf
EPS = 1e-9 # Protection division by zero


# Create Noisy Data for given set of Parameters

def Generate_Data(function, file_name: str, noise = 0.2, t_min: float = 0, t_max: float = 15, num = 1000):
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
    time = np.linspace(t_min, t_max, num)
    f_points = function(time)
    f_error = noise * f_points + EPS
    f_noisy = f_points + np.random.normal(0, np.abs(f_error), num)

    df = pd.DataFrame({
        'time': time,
        'datapoint': f_noisy
    })

    df.to_csv(f"Data_{file_name}.csv", index = False, float_format='%.6f')

    return time, f_points


# Generate_Data(gwf.Create_TimeMod_GW(0.3, 5, 10, 7.5),"Grav_Wave_TimeModded",
#               noise = 0.2, t_min = -7.5, t_max = 7.5, num = 1001)