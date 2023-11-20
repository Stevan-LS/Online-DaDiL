import numpy as np


def make_classification_dataset(mean,
                                cov,
                                v=None,
                                separation=1,
                                n=200):
    r"""Makes a classification dataset w/ Gaussian blobs

    Parameters
    ----------
    mean : np.array
        Mean vector for the first class
    cov : np.array
        Covariance matrix for the first class
    v : np.array, optional (default=None)
        Unit vector in the direction where classes will be separated.
        If None, draws a vector randomly from the unit sphere.
    separation : np.array, optional (default=1)
        Norm of the vector translating the two gaussian blobs.
    n : int, optional (default=200)
        Number of samples per class.
    """
    x1 = np.random.multivariate_normal(mean, cov, size=n)
    if v is None:
        v = np.random.randn(2,)
        v = separation * (v / np.linalg.norm(v)).reshape(1, -1)
    elif np.linalg.norm(v) != separation:
        v = separation * (v / np.linalg.norm(v)).reshape(1, -1)
    else:
        v = v.reshape(1, -1)
    x2 = x1 + v
    X = np.concatenate([x1, x2], axis=0)
    y = np.array([0] * len(x1) + [1] * len(x2))

    return X, y
