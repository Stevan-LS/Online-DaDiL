import torch
from pydil.ot_utils.pot_utils import emd
from pydil.ot_utils.pot_utils import unif
from pydil.ot_utils.sinkhorn import log_sinkhorn
from pydil.ot_utils.barycenters import wasserstein_barycenter


class WassersteinBarycenterTransport(torch.nn.Module):
    r"""Wasserstein Barycenter Transport, proposed by [1] and [2].

    This class implements the MSDA algorithm, called WBT, which,

    1. Calculates a Wasserstein barycenter of source domains

    2. Transports the Wasserstein barycenter to the target domain.

    Parameters
    ----------
    n_samples : int, optional (default=None)
        Number of samples in the barycentric domain.
        If not given, uses the sum of all source domain
        samples.
    reg_e : float, optional (default=0.0)
        Entropic regularization term. If bigger than 0,
        uses the Sinkhorn algorithm for calculating OT.
    n_iter_barycenter : int, optional (default=10)
        Number of iterations in the Wasserstein barycenter
        algorithm.
    n_iter_sinkhorn : int, optional (default=100)
        Number of iterations in the Sinkhorn algorithm.
        Only used if reg_e > 0.0.
    n_iter_emd : int, optional (default=1000000)
        Number of iterations in the Simplex algorithm.
        Only used if reg_e = 0.0
    tol : float, optional (default=1e-9)
        Stopping criterium for the Wasserstein barycenter
        algorithm.
    """
    def __init__(self,
                 n_samples=None,
                 reg_e=0.0,
                 n_iter_barycenter=10,
                 n_iter_sinkhorn=100,
                 n_iter_emd=1000000,
                 tol=1e-9):
        super(WassersteinBarycenterTransport, self).__init__()
        self.n_samples = n_samples
        self.reg_e = reg_e
        self.n_iter_barycenter = n_iter_barycenter
        self.n_iter_sinkhorn = n_iter_sinkhorn
        self.n_iter_emd = n_iter_emd
        self.tol = tol

    def forward(self, Xs, Ys, Xt, Yt=None):
        self.XB, self.YB, log = wasserstein_barycenter(
            XP=Xs,
            YP=Ys,
            XB=None,
            YB=None,
            weights=None,
            n_samples=self.n_samples,
            reg_e=self.reg_e,
            label_weight=None,
            n_iter_max=self.n_iter_barycenter,
            n_iter_sinkhorn=self.n_iter_sinkhorn,
            n_iter_emd=self.n_iter_emd,
            tol=self.tol,
            verbose=False,
            inner_verbose=False,
            propagate_labels=True,
            penalize_labels=True,
            log=True
        )

        self.transport_plans = log['transport_plans']

        uB = unif(self.XB)
        uT = unif(Xt)
        C_BT = torch.cdist(self.XB, Xt, p=2) ** 2

        with torch.no_grad():
            if self.reg_e > 0.0:
                pi_BT = log_sinkhorn(uB, uT, C_BT / C_BT.max(),
                                     reg_e=self.reg_e)
            else:
                pi_BT = emd(uB, uT, C_BT)
        self.transport_plans.append(pi_BT)
        T = pi_BT / pi_BT.sum(dim=1)[:, None]
        T[torch.isnan(T)] = 0.0
        TXB = torch.mm(T, Xt)

        return TXB, self.YB
