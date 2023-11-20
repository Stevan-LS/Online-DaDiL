import torch
import numpy as np
from pydil.toy_datasets.classification import make_classification_dataset


def gradual_da_toy_example(n=250, separation=6, angle=15,
                           angle_min=0, angle_max=45,
                           b_min=0, b_max=5, n_domains=10):
    r"""Creates a Toy example for Gradual domain adaptation.

    Parameters
    ----------
    n : int, optional (default=250)
        Number of samples per class in each domain.
    separation : float, optional (defaul=6)
        Norm of the separation vector between classes
    angle : float, optional (default=15)
        Angle for the unit vector separating classes
    angle_min : float, optional (default=0)
        Minimum angle in intermediate domains
    angle_max : float, optional (default=45)
        Maximum angle in intermediate domains
    b_min : float, optional (default=0)
        Minimum translation in intermediate domains
    b_max : float, optional (default=5)
        Maximum translation in intermediate domains
    n_domains : int, optional (default=10)
        Number of intermediate domains
    """
    # Creating datasets
    mu = np.array([0, 0])
    separation = 6

    angle = 15
    A = np.random.randn(2, 2)
    cov = .25 * np.dot(A.T, A) + np.eye(2)
    v = np.array([np.cos((np.pi / 180) * angle),
                  np.sin((np.pi / 180) * angle)])
    X, y = make_classification_dataset(mu,
                                       cov,
                                       v=v,
                                       separation=separation,
                                       n=n)

    Xs = torch.from_numpy(X).float()
    Ys = torch.nn.functional.one_hot(torch.from_numpy(y).long(),
                                     num_classes=2).float()

    angles = np.linspace(angle_min, angle_max, n_domains)[:-1]
    translation = np.linspace(b_min, b_max, n_domains)[:-1]

    intermediate_domains = []
    for θt, bt in zip(angles, translation):
        At = torch.Tensor([
            [np.cos((np.pi / 180) * θt), - np.sin((np.pi / 180) * θt)],
            [np.sin((np.pi / 180) * θt), np.cos((np.pi / 180) * θt)]
        ])
        Xi = Xs @ At + bt
        intermediate_domains.append(Xi)
    Xint = torch.stack(intermediate_domains)
    Yint = torch.stack([Ys for _ in range(len(Xi))])

    AT = torch.Tensor([
        [np.cos((np.pi / 180) * θt), - np.sin((np.pi / 180) * θt)],
        [np.sin((np.pi / 180) * θt), np.cos((np.pi / 180) * θt)]
    ])
    bT = b_max

    XT = Xs @ AT + bT
    YT = Ys.clone()

    return (Xs, Ys, Xint, Yint, XT, YT)
