import torch
import numpy as np
from pydil.toy_datasets.classification import make_classification_dataset


def msda_toy_example(n_datasets,
                     n_samples=400,
                     angle_min=0.0,
                     angle_max=45,
                     separation=6):
    mu = np.array([0, 0])
    angles = np.linspace(angle_min, angle_max, n_datasets)
    Xs, Ys = [], []
    for i in range(n_datasets - 1):
        A = np.random.randn(2, 2)
        cov = .25 * np.dot(A.T, A) + np.eye(2)
        v = np.array([np.cos((np.pi / 180) * angles[i]),
                      np.sin((np.pi / 180) * angles[i])])
        X, y = make_classification_dataset(mu, cov, v=v,
                                           separation=separation,
                                           n=n_samples)

        Xs.append(torch.from_numpy(X).float())
        Ys.append(
            torch.nn.functional.one_hot(torch.from_numpy(y).long(),
                                        num_classes=2).float())

    A = np.random.randn(2, 2)
    cov = .1 * np.dot(A.T, A) + np.eye(2)
    v = np.array([np.cos((np.pi / 180) * angles[-1]),
                  np.sin((np.pi / 180) * angles[-1])])
    Xt, yt = make_classification_dataset(mu, cov,
                                         v=v,
                                         separation=separation,
                                         n=n_samples)
    Xt = torch.from_numpy(Xt).float()
    Yt = torch.nn.functional.one_hot(
        torch.from_numpy(yt).long(), num_classes=2).float()

    return Xs, Ys, Xt, Yt
