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


def KLD_divergence_consecutive(x: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler divergence between a set of consecutive observations.
    Q is the first obsevation, used as a theory, a model, a description or an approximation of P.
    P is the second observation, representing the data or a measured probability distribution.
    """
    Q, P = x[:-1], x[1:]
    return P * (ma.log(P) - ma.log(Q))


def entropy(P: np.ndarray) -> np.ndarray:
    """
    Shannon entropy without summation.
    """
    return -P * (ma.log(P))
