import numpy as np
import numpy.ma as ma


def KLD_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
    P represents the data, the observations, or a measured probability distribution.
    Q represents a theory, a model, a description or an approximation of P.
    P - Q makes the regular KL divergence a distance complying with the triangle inequality.
    The distance is always > 0.
    """
    return (P - Q) * (ma.log(P) - ma.log(Q))


def KLD_distance_overused(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
    P represents the data, the observations, or a measured probability distribution.
    Q represents a theory, a model, a description or an approximation of P.
    P - Q makes the regular KL divergence a distance complying with the triangle inequality.
    np.where(P < Q, -1, 1) adds a minus sign if P < Q
    """
    return np.where(P < Q, -1, 1) * (P - Q) * (np.log(P / Q))


def KLD_distance_consecutive(x: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance between a set of consecutive observations.
    """
    Q, P = x[:-1], x[1:]
    return (P - Q) * (np.log(P) - np.log(Q))


def KLD_divergence_consecutive(x: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler divergence between a set of consecutive observations.
    """
    Q, P = x[:-1], x[1:]
    return np.abs(P * (np.log(P) - np.log(Q)))


def KLD_divergence(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler divergence between a set of consecutive observations.
    """
    return np.abs(P * (ma.log(P) - ma.log(Q)))


def entropy(P: np.ndarray) -> np.ndarray:
    """
    Shannon entropy without summation.
    """
    return -P * (ma.log(P))
