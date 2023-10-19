import torch


def sqrtm(A, return_inv=False):
    D, V = torch.linalg.eig(A)

    A_sqrt = torch.mm(V, torch.mm(torch.diag(D.pow(1 / 2)), V.T)).real
    if return_inv:
        A_sqrt_neg = torch.mm(V, torch.mm(torch.diag(D.pow(-1 / 2)), V.T)).real
        return A_sqrt, A_sqrt_neg
    return A_sqrt


def proj_spd(A, reg=1e-6):
    _A = 0.5 * (A + A.T)
    D, V = torch.linalg.eig(_A)
    Ddiag = torch.diag(D.real)
    Ddiag = torch.maximum(Ddiag, reg * torch.ones_like(Ddiag))
    return torch.mm(V.real, torch.mm(Ddiag, V.real.T)).real
