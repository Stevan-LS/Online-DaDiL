r"""Optimal Transport Mapping module.

In this module, we implement various techniques
for estimating an Optimal Transport mapping, among
which,

- Linear Mapping estimation, by assuming distributions
are Gaussians

- Barycentric projection, which uses the Kantorovich
formulation.

"""

import torch

from pydil.ot_utils.pot_utils import emd
from pydil.ot_utils.pot_utils import unif
from pydil.ot_utils.sinkhorn import log_sinkhorn


class BarycentricMapping(torch.nn.Module):
    def __init__(self, reg=0.0, num_iter_sinkhorn=50):
        super(BarycentricMapping, self).__init__()

        self.reg = reg
        self.num_iter_sinkhorn = num_iter_sinkhorn

    def fit(self, XP, XQ, p=None, q=None):
        # if p is not given, assume uniform
        if p is None:
            self.p = unif(len(XP))
        else:
            self.p = p

        # if q is not given, assume uniform
        if q is None:
            q = unif(len(XQ))

        # Calculates pairwise distances
        C = torch.cdist(XP, XQ, p=2) ** 2

        # Calculates transport plan
        with torch.no_grad():
            if self.reg > 0.0:
                self.π = log_sinkhorn(self.p, q, C,
                                      reg_e=self.reg,
                                      n_iter_max=self.num_iter_sinkhorn)
            else:
                self.π = emd(self.p, q, C)

    def forward(self, XP, XQ, p=None, q=None):
        # Defines internal π
        self.fit(XP, XQ, p, q)

        return torch.mm((self.π / self.p[:, None]), XQ)


class LinearOptimalTransportMapping(torch.nn.Module):
    def __init__(self, reg=1e-6):
        super(LinearOptimalTransportMapping, self).__init__()
        self.reg = reg
        self.fitted = False

    def fit(self, XP, XQ):
        with torch.no_grad():
            self.mP = torch.mean(XP, dim=0, keepdim=True)
            self.mQ = torch.mean(XQ, dim=0, keepdim=True)

            self.sP = torch.cov(XP.T) + self.reg * torch.eye(XP.shape[1])
            self.sQ = torch.cov(XQ.T) + self.reg * torch.eye(XQ.shape[1])

            D, V = torch.linalg.eig(self.sP)
            sP_pos = torch.mm(V, torch.mm(torch.diag(D.pow(1 / 2)), V.T)).real
            sP_neg = torch.mm(V, torch.mm(torch.diag(D.pow(-1 / 2)), V.T)).real

            self.M = torch.mm(sP_pos, torch.mm(self.sQ, sP_pos))
            self.A = torch.mm(sP_neg, torch.mm(self.M, sP_neg))
        self.fitted = True

    def dist(self):
        mean_dist = torch.dist(self.mP, self.mQ, p=2) ** 2
        bures_metric = (torch.trace(self.sP) +
                        torch.trace(self.sQ) -
                        2 * torch.trace(self.M))
        return torch.sqrt(mean_dist ** 2 + bures_metric)

    def forward(self, XP, XQ, p=None, q=None):
        if not self.fitted:
            self.fit(XP, XQ)

        return self.mQ + torch.mm(XP - self.mP, self.A)
