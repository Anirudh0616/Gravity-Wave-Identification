import numpy as np


def Create_Problem_GW(alpha, beta, gamma):
    """
    Model Function of Gravitational Wave given in Problem Statement

    :param alpha: Parameter Alpha (0, 2)
    :param beta: Parameter Beta (1, 10)
    :param gamma: Parameter Gamma (1, 20)
    :return: Gravitational Wave Function for given Params
    """
    def Gravitational_Wave(t):
        return alpha * np.exp(t) * (1 - np.tanh(2 * (t - beta))) * np.sin(gamma * t)

    return Gravitational_Wave

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

def likelihood(y_data: np.ndarray, y_prior: np.ndarray):
    y_err = 0.2 * np.max(y_data)
    Y = np.sum((y_data - y_prior) ** 2) / y_err
    return -0.5 * Y