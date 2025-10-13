import numpy as np


class Gravitation_Wave:
    @staticmethod
    def Time_series(alpha, beta, gamma):
        """
        Returns Time Series function which takes input as time
        :param alpha: Parameter Alpha (0, 2)
        :param beta: Parameter Beta (1, 10)
        :param gamma: Parameter Gamma (1, 20)
        :return: function(t)
        """
        def function(t):
            return alpha * np.exp(t) * (1 - np.tanh(2 * (t - beta))) * np.sin(gamma * t)
        return function

    @staticmethod
    def Parameter_Space( t):
        """
        Returns Parameter Space function which takes input as Parameter
        :param t: time series data
        :return: function(alpha, beta, gamma)
        """
        def function(alpha, beta, gamma):
            return alpha * np.exp(t) * (1 - np.tanh(2 * (t - beta))) * np.sin(gamma * t)
        return function

def Create_TimeMod_GW(alpha, beta, gamma, t_mean):
    """
    Model Function of Time-Mod Gravitational Wave
    :param alpha: Parameter Alpha (0, 2)
    :param beta: Parameter Beta (1, 10)
    :param gamma: Parameter Gamma (1, 20)
    :param t_mean: Mean of Time series
    :return: Gravitational Wave Function for given Params
    """
    def Gravitational_Wave(t):
        return alpha * np.exp(t) * (1 - np.tanh(2 * (t - beta + t_mean))) * np.sin(gamma * t)

    return Gravitational_Wave

def likelihood_reduced(y_data: np.ndarray, y_prior: np.ndarray):
    y_err = 0.2 * np.std(y_data)
    Y = np.mean((y_data - y_prior) ** 2) / y_err**2
    return -0.5 * Y

def likelihood(y_data: np.ndarray, y_prior: np.ndarray):
    y_err = 0.1 * (y_data + y_prior) + 1e-6
    Y = np.sum(((y_data - y_prior)/y_err) ** 2 )
    return -0.5 * Y