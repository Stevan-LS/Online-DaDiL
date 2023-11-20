r"""Utilities for Linear Algebra."""

import torch


def sqrtm(A, return_inv=False):
    r"""Calculates the square root matrix of a symmetric matrix A
    through Eigendecomposition. This is useful, for instance, when
    A is a covariance matrix. Formally, the square root B of A is a
    matrix such that :math:`A = BB`. Let :math:`D` and :math:`V` be
    matrices such that :math:`A = VDV^{-1}`. The square-root B of A
    can be defined as,

    .. math::
        B = VD^{1/2}V^{-1}


    Parameters
    ----------
    A : tensor
        Matrix of shape (n, n).
    return_inv : bool, optional (default=False)
        If True, returns the inverse of B.
    """
    D, V = torch.linalg.eig(A)

    A_sqrt = torch.mm(V, torch.mm(torch.diag(D.pow(1 / 2)), V.T)).real
    if return_inv:
        A_sqrt_neg = torch.mm(V, torch.mm(torch.diag(D.pow(-1 / 2)), V.T)).real
        return A_sqrt, A_sqrt_neg
    return A_sqrt


def proj_spd(A, reg=1e-6):
    r"""Projects a square matrix A in the manifold of
    Symmetric Positive-Definite (SPD) matrices.

    Parameters
    ----------
    A : tensor
        Matrix of shape (n, n)
    reg : float, optional (default=1e-6)
        Minimum value for eigenvalues of A.
    """
    _A = 0.5 * (A + A.T)
    D, V = torch.linalg.eig(_A)
    Ddiag = torch.diag(D.real)
    Ddiag = torch.maximum(Ddiag, reg * torch.ones_like(Ddiag))
    return torch.mm(V.real, torch.mm(Ddiag, V.real.T)).real
