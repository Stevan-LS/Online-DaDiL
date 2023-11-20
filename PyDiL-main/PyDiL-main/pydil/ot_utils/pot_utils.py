import ot
import torch
import numpy as np
from pydil.linalg_utils.matrices import sqrtm
from pydil.ipms.parametric_ipms import parametric_bures_wasserstein_metric


def proj_simplex(v, z=1):
    r"""Re-implements ot.utils.proj_simplex using torch.
    This was necessary due to strange behavior of POT w.r.t. GPU memory."""
    n = v.shape[0]
    if v.ndim == 1:
        d1 = 1
        v = v[:, None]
    else:
        d1 = 0
    d = v.shape[1]

    # sort u in ascending order
    u, _ = torch.sort(v, dim=0)
    # take the descending order
    u = torch.flip(u, dims=[0])
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(n, dtype=v.dtype)[:, None] + 1
    cond = u - cssv / ind > 0
    rho = torch.sum(cond, 0)
    theta = cssv[rho - 1, torch.arange(d)] / rho
    w = torch.maximum(v - theta[None, :], torch.zeros(v.shape, dtype=v.dtype))
    if d1:
        return w[:, 0]
    else:
        return w


def unif(n, device='cpu', dtype=torch.float32):
    r"""Returns uniform sample weights for a number of samples $n > 0$.

    Parameters
    ----------
    n : int
        Number of samples
    device : str, optional (default='cpu')
        Whether the returned tensor is on 'cpu' or 'gpu'.
    dtype : torch dtype, optional (default=torch.float32)
        Data type for the returned vector.
    """
    return torch.ones(n, device=device, dtype=dtype) / n


def emd(a, b, C, n_iter_max=100000):
    r"""Wrapper function for ```ot.emd```. NOTE: This function
    receives torch tensors and converts them to numpy, before
    calling the Earth Mover Distance function of POT.

    Parameters
    ----------
    a : tensor
        Tensor of shape (n,) containing the importance of each sample in the
        distribution P. Must be positive and sum to one.
    b : tensor
        Tensor of shape (m,) containing the importance of each sample in the
        distribution Q. Must be positive and sum to one.
    C : tensor
        Tensor of shape (n, m) containing the pairwise distance between samples
        of P and Q.
    n_iter_max : int, optional (default=1000000)
        Number of iterations of Linear Programming.
    """
    _a = a.detach().cpu().numpy()
    _b = b.detach().cpu().numpy()
    _C = C.detach().cpu().numpy()

    np_ot_plan = ot.emd(_a, _b, _C, numItermax=n_iter_max)

    return torch.from_numpy(np_ot_plan).to(C.dtype).to(C.device)


def emd2(a, b, C, n_iter_max=100000):
    r"""Wrapper function for ```ot.emd2```. NOTE: This function
    receives torch tensors and converts them to numpy, before
    calling the Earth Mover Distance function of POT.

    Parameters
    ----------
    a : tensor
        Tensor of shape (n,) containing the importance of each sample in the
        distribution P. Must be positive and sum to one.
    b : tensor
        Tensor of shape (m,) containing the importance of each sample in the
        distribution Q. Must be positive and sum to one.
    C : tensor
        Tensor of shape (n, m) containing the pairwise distance between samples
        of P and Q.
    n_iter_max : int, optional (default=1000000)
        Number of iterations of Linear Programming.
    """
    dtype = C.dtype
    device = C.device

    _a = a.detach().cpu().numpy()
    _b = b.detach().cpu().numpy()
    _C = C.detach().cpu().numpy()

    ot_plan = torch.from_numpy(
        ot.emd(_a, _b, _C, numItermax=n_iter_max)).to(dtype).to(device)

    return torch.sum(ot_plan * C)


def gmm_emd(means_s,
            covs_s,
            means_t,
            covs_t,
            weights_s=None,
            weights_t=None,
            cov_type='full'):
    ns, nt = means_s.shape[0], means_t.shape[0]

    C = np.zeros([ns, nt])
    for cs in range(ns):
        for ct in range(nt):
            C[cs, ct] = parametric_bures_wasserstein_metric(means_s[cs],
                                                            covs_s[cs],
                                                            means_t[ct],
                                                            covs_t[ct],
                                                            cov_type=cov_type,
                                                            item=True)
    if weights_s is None:
        weights_s = np.ones(ns) / ns
    if weights_t is None:
        weights_t = np.ones(nt) / nt
    ot_plan = ot.emd(weights_s, weights_t, C)

    return ot_plan


def gmm_emd2(means_s,
             covs_s,
             means_t,
             covs_t,
             weights_s=None,
             weights_t=None,
             cov_type='full'):
    ns, nt = means_s.shape[0], means_t.shape[0]

    C = torch.zeros([ns, nt])
    for cs in range(ns):
        for ct in range(nt):
            C[cs, ct] = parametric_bures_wasserstein_metric(means_s[cs],
                                                            covs_s[cs],
                                                            means_t[ct],
                                                            covs_t[ct],
                                                            cov_type=cov_type,
                                                            item=False)
    with torch.no_grad():
        if weights_s is None:
            weights_s = np.ones(ns) / ns
        if weights_t is None:
            weights_t = np.ones(nt) / nt
        ot_plan = ot.emd(weights_s, weights_t, C.cpu().numpy())
    ot_plan = torch.from_numpy(ot_plan).to(C.dtype).to(C.device)

    return torch.sum(ot_plan * C)


def map_gaussians(mean_s, cov_s, mean_t, cov_t, cov_type='full'):
    if cov_type == 'full':
        sqrtm_cov_s_pos, sqrtm_cov_s_neg = sqrtm(cov_s, return_inv=True)
        M = sqrtm(sqrtm_cov_s_pos @ cov_t @ sqrtm_cov_s_pos)
        A = sqrtm_cov_s_neg @ M @ sqrtm_cov_s_neg
    else:
        A = torch.diag(cov_t / cov_s)
    b = mean_t - mean_s @ A

    return A, b
