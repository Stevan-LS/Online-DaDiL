r"""Sinkhorn algorithm module. This module implements the
Sinkhorn algorithm of [1]

[1] Cuturi, M. (2013). Sinkhorn distances: Lightspeed
    computation of optimal transport. Advances in neural
    information processing systems, 26.

"""

import torch


def sinkhorn(a, b, C,
             reg_e=1.0,
             n_iter_max=100,
             threshold=1e-9,
             verbose=False,
             return_dual_vars=False):
    r"""Sinkhorn Algorithm. Solves a regularized version of the Optimal
    Transport by problem matrix scaling. The optimization problem consists on

    .. math::
        \pi = argmin \langle \pi, C \rangle_{F} - \epsilon H(\pi)

    under the constraints :math:`\sum_{j=1}^{m}\pi_{ij} = a_{i}` and
    :math:`\sum_{i=1}^{n}\pi_{ij} = b_{j}`. The entropic term :math:`H(\pi)` is
    defined as,

    .. math::
        H(\pi) = -\sum_{i=1}^{n}\sum_{j=1}^{m}\pi_{ij} \log \pi_{ij}


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
    reg_e : float, optional (default=1.0)
        Entropic regularization penalty.
    threshold : float, optional (default=1e-9)
        Tolerance of error in the Sinkhorn updates.
    verbose : bool, optional (default=False)
        If True, displays information about Sinkhorn iterations.
    return_dual_vars : bool (default=False)
        If True, returns dual solution of OT problem.
    """
    device = C.device
    dtype = C.dtype

    u = torch.ones(len(a), device=device, dtype=dtype) / len(a)
    v = torch.ones(len(b), device=device, dtype=dtype) / len(b)
    K = torch.exp(- C / reg_e)

    for it in range(n_iter_max):
        _u = u.clone()
        _v = v.clone()

        u = a / torch.matmul(K, v)
        v = b / torch.matmul(K.T, u)

        err_u = (u - _u).abs().sum()
        err_v = (v - _v).abs().sum()
        err = (err_u + err_v) / 2
        if verbose:
            print('it {}, err_u: {}, err_v: {}'.format(it, err_u, err_v))
        if (err < threshold).data.cpu().numpy():
            break

    pi = torch.mm(torch.diag(u), torch.mm(K, torch.diag(v)))
    if return_dual_vars:
        return u, v, pi
    return pi


def unbalanced_sinkhorn(a, b, C,
                        reg_e=1.0,
                        reg_m=1.0,
                        n_iter_max=100,
                        threshold=1e-9,
                        verbose=False,
                        return_dual_vars=False):
    r"""Unbalanced Sinkhorn Algorithm. Solves a regularized version of the
    Optimal Transport by problem matrix scaling. The optimization problem
    consists on

    .. math::
        \pi = argmin \langle \pi, C \rangle_{F} - \epsilon H(\pi) +
        \tau KL(a||\pi \mathbf{1}_{m}) + \tau KL(b||\mathbf{1}_{n} \pi)

    under the constraints :math:`\sum_{j=1}^{m}\pi_{ij} = a_{i}` and
    :math:`\sum_{i=1}^{n}\pi_{ij} = b_{j}`. The entropic term :math:`H(\pi)` is
    defined as,

    .. math::
        H(\pi) = -\sum_{i=1}^{n}\sum_{j=1}^{m}\pi_{ij} \log \pi_{ij}


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
    reg_e : float, optional (default=1.0)
        Entropic regularization penalty.
    reg_m : float, optional (default=1.0)
        Unbalanced regularization penalty.
    threshold : float, optional (default=1e-9)
        Tolerance of error in the Sinkhorn updates.
    verbose : bool, optional (default=False)
        If True, displays information about Sinkhorn iterations.
    return_dual_vars : bool (default=False)
        If True, returns dual solution of OT problem.
    """
    device = C.device
    dtype = C.dtype

    u = torch.ones(len(a), device=device, dtype=dtype) / len(a)
    v = torch.ones(len(b), device=device, dtype=dtype) / len(b)
    K = torch.exp(- C / reg_e)

    exp = reg_m / (reg_m + reg_e)

    for it in range(n_iter_max):
        _u = u.clone()
        _v = v.clone()

        u = (a / torch.matmul(K, v)) ** exp
        v = (b / torch.matmul(K.T, u)) ** exp

        err_u = (u - _u).abs().sum()
        err_v = (v - _v).abs().sum()
        err = (err_u + err_v) / 2
        if verbose:
            print('it {}, err_u: {}, err_v: {}'.format(it, err_u, err_v))
        if (err < threshold).data.cpu().numpy():
            break

    pi = torch.mm(torch.diag(u), torch.mm(K, torch.diag(v)))
    if return_dual_vars:
        return u, v, pi
    return pi


def log_sinkhorn(a, b, C,
                 reg_e=1.0,
                 n_iter_max=100,
                 threshold=1e-9,
                 verbose=False,
                 return_dual_vars=False):
    r"""Sinkhorn algorithm where updates are done in log-space.

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
    reg_e : float, optional (default=1.0)
        Entropic regularization penalty.
    threshold : float, optional (default=1e-9)
        Tolerance of error in the Sinkhorn updates.
    verbose : bool, optional (default=False)
        If True, displays information about Sinkhorn iterations.
    return_dual_vars : bool (default=False)
        If True, returns dual solution of OT problem.
    """
    device = C.device
    dtype = C.dtype

    f = torch.ones(len(a), device=device, dtype=dtype) / len(a)
    g = torch.ones(len(b), device=device, dtype=dtype) / len(b)

    for it in range(n_iter_max):
        _f = f.clone()

        logM = (- C + f[:, None] + g[None, :]) / reg_e
        f = f + reg_e * (torch.log(a) - torch.logsumexp(logM, dim=1))
        logM = (- C + f[:, None] + g[None, :]) / reg_e
        g = g + reg_e * (torch.log(b) - torch.logsumexp(logM, dim=0))

        err = (f - _f).abs().sum()
        if verbose:
            print('it {}, err: {}'.format(it, err))
        if (err < threshold).data.cpu().numpy():
            break

    u = torch.exp(f / reg_e)
    v = torch.exp(g / reg_e)
    pi = torch.exp((-C + f[:, None] + g[None, :]) / reg_e)
    if return_dual_vars:
        return u, v, pi
    return pi


def log_unbalanced_sinkhorn(a, b, C,
                            reg_e=1.0,
                            reg_m=1.0,
                            n_iter_max=100,
                            threshold=1e-9,
                            verbose=False,
                            return_dual_vars=False):
    r"""Unbalanced Sinkhorn algorithm where updates are done in log-space.

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
    reg_e : float, optional (default=1.0)
        Entropic regularization penalty.
    reg_m : float, optional (default=1.0)
        Unbalanced regularization penalty.
    threshold : float, optional (default=1e-9)
        Tolerance of error in the Sinkhorn updates.
    verbose : bool, optional (default=False)
        If True, displays information about Sinkhorn iterations.
    return_dual_vars : bool (default=False)
        If True, returns dual solution of OT problem.
    """
    device = C.device
    dtype = C.dtype

    f = torch.ones(len(a), device=device, dtype=dtype) / len(a)
    g = torch.ones(len(b), device=device, dtype=dtype) / len(b)

    coef = reg_e * (reg_m / (reg_m + reg_e))

    for it in range(n_iter_max):
        _f = f.clone()

        logM = (- C + f[:, None] + g[None, :]) / reg_e
        f = f + coef * (torch.log(a) - torch.logsumexp(logM, dim=1))
        logM = (- C + f[:, None] + g[None, :]) / reg_e
        g = g + coef * (torch.log(b) - torch.logsumexp(logM, dim=0))

        err = (f - _f).abs().sum()
        if verbose:
            print('it {}, err: {}'.format(it, err))
        if (err < threshold).data.cpu().numpy():
            break

    u = torch.exp(f / reg_e)
    v = torch.exp(g / reg_e)
    pi = torch.exp((-C + f[:, None] + g[None, :]) / reg_e)
    if return_dual_vars:
        return u, v, pi
    return pi
