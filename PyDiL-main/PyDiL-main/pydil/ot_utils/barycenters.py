r"""Module for computing Wasserstein barycenters

[1] Agueh, M., & Carlier, G. (2011). Barycenters in the Wasserstein space.
    SIAM Journal on Mathematical Analysis, 43(2), 904-924.

[2] Cuturi, M., & Doucet, A. (2014, June). Fast computation of Wasserstein
    barycenters. In International conference on machine learning
    (pp. 685-693). PMLR.

[3] Álvarez-Esteban, P. C., Del Barrio, E., Cuesta-Albertos, J. A., &
    Matrán, C. (2016). A fixed-point approach to barycenters in
    Wasserstein space. Journal of Mathematical Analysis and Applications,
    441(2), 744-762.

[4] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal
    transport. In Advances in neural information processing systems, 26.
"""

import time
import torch
import numpy as np

from pydil.ot_utils.pot_utils import emd
from pydil.ot_utils.pot_utils import unif
from pydil.linalg_utils.matrices import sqrtm
from pydil.ot_utils.sinkhorn import log_sinkhorn


def label2hist(Y):
    y = Y.argmax(dim=1)
    u, c = torch.unique(y, return_counts=True)
    hist = torch.zeros(Y.shape[1])
    hist[u] += c
    return hist


def wasserstein_barycenter(XP,
                           YP=None,
                           XB=None,
                           YB=None,
                           weights=None,
                           n_samples=None,
                           reg_e=0.0,
                           label_weight=None,
                           n_iter_max=100,
                           n_iter_sinkhorn=1000,
                           n_iter_emd=100000,
                           tol=1e-4,
                           verbose=False,
                           inner_verbose=False,
                           propagate_labels=False,
                           penalize_labels=False,
                           log=False):
    r"""Computes the Wasserstein Barycenter [1] for a list of distributions
    :math:`\mathcal{P}`, containing :math:`\hat{P}_{1}, \cdots ,\hat{P}_{K}`
    and weights :math:`\alpha \in \Delta_{K}`. Each distribution is
    parametrized through their support
    :math:`\mathbf{X}^{( P_{k} )}, k=1, \cdots ,K`. This consists on a
    implementation of the Free-Support Wasserstien Barycenter of [2]. Our
    implementation relies on the fixed-point iteration of [3],

    .. math::
        \hat{B}^{(it+1)} = \psi( \hat{B}^{(it)} ),

    where :math:`\psi(\hat{P}) = T_{it,\sharp}\hat{P}`,
    :math:`T_{it} = \sum_{k}\alpha_{k}T_{k,it}`, for :math:`T_{k,it}`,
    the  barycentric mapping between :math:`\hat{P}_{k}` and
    :math:`\hat{B}^{(it)}`.

    Parameters
    ----------
    XP : List of tensors
        List of tensors of shape (nk, d) with the features of the support of
        each distribution Pk.
    YP : List of tensors, optional (default=None)
        List of tensors of shape (nk, nc) with the labels of the support of
        each distribution Pk.
    XB : tensor, optional (default=None)
        Tensor of shape (n, d) with the initialization for the features of
        the barycenter support.
    YB : tensor, optional (default=None)
        Tensor of shape (n, d) with the initialization for the labels of
        the barycenter support.
    weights : tensor, optional (default=None)
        Weight of each distribution in (XP, YP). It is a tensor of shape
        (K,), whose components are all positive and it sums to one.
    n_samples : int, optional (default=None)
        Number of samples in the barycenter support. Only used if (XB, YB)
        were not given.
    reg_e : float, optional (default=0.0)
        Entropic regularization. If reg_e > 0.0 uses the Sinkhorn algorithm
        for computing the OT plans.
    label_weight : float, optional (default=None)
        Weight for the label metric. It is described as beta in the main paper.
        If None is given, uses beta as the maximum pairwise distance between
        samples of P and Q.
    n_iter_max : int, optional (default=100)
        Maximum number of iterations of the Barycenter algorithm.
    n_iter_sinkhorn : int, optional (default=1000)
        Maximum number of iterations of the Sinkhorn algorithm. Only used for
        reg_e > 0.0.
    n_iter_emd : int, optional (default=1000000)
        Maximum number of iterations for Linear Programming. Only used if
        reg_e = 0.0.
    tol : float, optional (default=1e-4)
        Tolerance for the iterations of the Wasserstein barycenter algorithm.
        If a given update does not change the objective function by a value
        superior to tol, the algorithm halts.
    verbose : bool, optional (default=False)
        If True, prints information about barycenter iterations.
    inner_verbose : bool, optional (default=False)
        If True, prints information about Sinkhorn iterations.
    propagate_labels : bool, optional (default=False)
        If True, propagates labels from distributions P towards the barycenter.
        Only used if YP was given.
    penalize_labels : bool, optional (default=False)
        If True, penalizes OT plans that mix classes.
    log : bool, optional (default=False)
        If True, returns log information about barycenter iterations.
    """
    if YP is None:
        if propagate_labels:
            raise ValueError(("Expected labels to be given in 'y'"
                             " for 'propagate_labels' = True"))
        if penalize_labels:
            raise ValueError(("Expected labels to be given in 'y'"
                              " for 'penalize_labels' = True"))
    dtype = XP[0].dtype
    device = XP[0].device

    if n_samples is None and XB is None:
        # If number of points is not provided,
        # assume that the support of the barycenter
        # has sum(nsi) where si is the i-th source
        # domain.
        n_samples = int(np.sum([len(XPk) for XPk in XP]))

    if weights is None:
        weights = unif(len(XP), device=device, dtype=dtype)
    else:
        # assert (weights > 0.0).all()
        # assert torch.isclose(weights.sum(), torch.tensor([1.0])).item()
        pass

    it = 0
    comp_start = time.time()

    if XB is None:
        XB = torch.randn(n_samples, XP[0].shape[1])

    if YP is not None and YB is None:
        with torch.no_grad():
            count_labels = [
                label2hist(YPk).numpy()
                for YPk in YP
            ]
            prob_labels = np.stack([c_k / c_k.sum()
                                    for c_k in count_labels])
            prob_labels = np.einsum('i,ij->j', weights, prob_labels)
            prob_labels /= prob_labels.sum()
            _labels = np.arange(YP[0].shape[1])
            YB = torch.nn.functional.one_hot(
                torch.from_numpy(
                    np.random.choice(_labels, size=n_samples,
                                     replace=True, p=prob_labels)
                ).long(), num_classes=YP[0].shape[1]
            ).float()

    # Displacement of points in the support
    delta = tol + 1
    last_loss = np.inf
    # Create uniform weights
    u_P = [unif(len(XPk), device=device) for XPk in XP]
    u_B = unif(len(XB), device=device)

    if verbose:
        print("-" * (26 * 4 + 1))
        print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration',
                                                     'Loss',
                                                     'δLoss',
                                                     'Elapsed Time'))
        print("-" * (26 * 4 + 1))

    if log:
        extra_ret = {'transport_plans': [], 'd_loss': [], 'loss_hist': []}

    while (delta > tol and it < n_iter_max):
        # NOTE: Here we solve the barycenter problem without calculating
        # gradients at each iteration, as per the envelope theorem, we
        # only need to compute gradients at optimality.
        with torch.no_grad():
            tstart = time.time()
            C, ot_plans = [], []

            for k in range(len(XP)):
                C_k = torch.cdist(XP[k], XB, p=2) ** 2
                if label_weight is None:
                    _lw = C_k.max()
                else:
                    _lw = label_weight
                if penalize_labels:
                    C_k = C_k + _lw * torch.cdist(YP[k], YB, p=2) ** 2
                C.append(C_k)
                if reg_e > 0.0:
                    plan_k = log_sinkhorn(u_P[k], u_B, C_k / C_k.max(),
                                          reg_e=reg_e,
                                          n_iter_max=n_iter_sinkhorn,
                                          verbose=inner_verbose)
                else:
                    plan_k = emd(u_P[k], u_B, C_k, n_iter_max=n_iter_emd)
                ot_plans.append(plan_k.to(dtype))
            XB = sum([
                w_k * n_samples * torch.mm(plan_k.T, XP_k)
                for w_k, plan_k, XP_k in zip(weights, ot_plans, XP)
            ])
            if propagate_labels:
                YB = sum([
                    w_k * n_samples * torch.mm(plan_k.T, YP_k)
                    for w_k, plan_k, YP_k in zip(weights, ot_plans, YP)
                ])
            loss = sum([
                torch.sum(C_k * plan_k) for C_k, plan_k in zip(C, ot_plans)
            ])
            delta = torch.norm(loss - last_loss) / n_samples
            last_loss = loss
            tfinish = time.time()

            if verbose:
                delta_t = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it,
                                                             loss,
                                                             delta,
                                                             delta_t))

            if log:
                extra_ret['loss_hist'].append(loss)
                extra_ret['d_loss'].append(delta)

            it += 1
    if verbose:
        duration = time.time() - comp_start
        print("-" * (26 * 4 + 1))
        print(f"Barycenter calculation took {duration} seconds")
    # Re-evaluate the support at optimality for calculating the gradients
    # NOTE: now we define the support while holding its gradients w.r.t. the
    # weight vector and eventually the support.
    XB = sum([
        w_k * n_samples * torch.mm(plan_k.T, XP_k)
        for w_k, plan_k, XP_k in zip(weights, ot_plans, XP)
    ])
    if propagate_labels:
        YB = sum([
            w_k * n_samples * torch.mm(plan_k.T, YP_k)
            for w_k, plan_k, YP_k in zip(weights, ot_plans, YP)
        ])
        if log:
            extra_ret['transport_plans'] = ot_plans
            return XB, YB, extra_ret
        return XB, YB
    if log:
        extra_ret['transport_plans'] = ot_plans
        return XB, extra_ret
    return XB


def bures_wasserstein_barycenter(means,
                                 covs,
                                 weights=None,
                                 cov_type='full',
                                 n_iter_max=10,
                                 tol=1e-6,
                                 verbose=False,
                                 extra_ret=False):
    r"""Computes the Bures-Wasserstein barycenter of
    means and covariances of Gaussian distributions.

    Parameters
    ----------
    means : tensor
        Tensor of shape (K, d), where K is the number of
        distributions in the barycenter calculation.
    covs : tensor
        Tensor of shape (K, d, d), or (K, d), for 'full'
        or 'diag' covariance matrices.
    weights : tensor, optional (default=None)
        Tensor of shape (K,) with positive entries that
        sum to one. If not given, considers uniform weights.
    cov_type : str, optional (default='full')
        Either 'full', 'commute' or 'diag' for the kind of
        covariance matrix of distributions P.
    n_iter_max : int, optional (default=10)
        Number of fixed-point iterations for determining the
        covariance matrix of the barycenter, when cov_type = 'full'.
    verbose : bool, optional (default=False)
        If True, prints information about fixed-point iterations.
    extra_ret : bool, optional (default=False)
        If True, returns additional information about fixed-point iterations.
    """
    dtype = means.dtype
    device = means.device

    if weights is None:
        weights = unif(means.shape[0], device=device, dtype=dtype)
    else:
        # assert (weights > 0.0).all()
        # assert torch.isclose(weights.sum(), torch.tensor([1.0])).item()
        pass

    mB = torch.einsum('i,ij->j', weights, means)

    if cov_type.lower() == 'diag':
        covB = torch.diag(torch.einsum('i,ij->j', weights, covs))
    elif cov_type.lower() == 'commute':
        s_covs = []
        for cov_i in covs:
            s_covs.append(sqrtm(cov_i))
        sqrtm_covB = sum([
            w_i * s_cov_i for w_i, s_cov_i in zip(weights, s_covs)
        ])
        covB = torch.mm(sqrtm_covB, sqrtm_covB.T)
    else:
        it = 0
        delta = np.inf

        covB = torch.eye(len(mB)).to(means[0].dtype)
        _covB = covB.clone()

        if extra_ret:
            history = [covB]

        while delta > tol and it < n_iter_max:
            std_pos = sqrtm(covB)
            std_neg = torch.linalg.inv(std_pos)

            inner = sum([
                w_k * sqrtm(
                    torch.matmul(std_pos, torch.matmul(cov_k, std_pos))
                )
                for w_k, cov_k in zip(weights, covs)
            ])
            inner = torch.matmul(inner, inner)

            covB = torch.matmul(std_neg, torch.matmul(inner, std_neg))

            if extra_ret:
                history.append(covB)

            it += 1
            delta = torch.linalg.norm(covB - _covB, ord='fro')

            _covB = covB.clone()

        if verbose:
            print("It {}, delta {}".format(it, delta))
    if extra_ret:
        return mB, covB, history
    return mB, covB
