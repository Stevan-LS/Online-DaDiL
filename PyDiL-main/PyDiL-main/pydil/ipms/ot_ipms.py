r"""Module for losses between probability distributions: $\mathcal{L}(P,Q)$."""

import ot
import torch

from pydil.ot_utils.pot_utils import emd
from pydil.ot_utils.pot_utils import unif
from pydil.ot_utils.sinkhorn import log_sinkhorn
from pydil.ot_utils.sinkhorn import log_unbalanced_sinkhorn


def bures_wasserstein_metric(XP, XQ, cov_type='full', reg=1e-6):
    meanP = torch.mean(XP, dim=0)
    meanQ = torch.mean(XQ, dim=0)
    mean_diff = torch.linalg.norm(meanP - meanQ) ** 2

    if cov_type == 'full':
        covP = torch.cov(XP.T) + reg * torch.eye(XP.shape[1])
        covQ = torch.cov(XQ.T) + reg * torch.eye(XQ.shape[1])

        DP, VP = torch.linalg.eig(covP)
        DP = torch.diag(DP ** (1 / 2))
        covP_sqrt = (VP @ DP @ VP.T).real

        D, V = torch.linalg.eig(covP_sqrt @ covQ @ covP_sqrt)
        D = torch.diag(D ** (1 / 2))
        M = (V @ D @ V.T).real

        cov_diff = torch.trace(covP) + torch.trace(covQ) - 2 * torch.trace(M)
    elif cov_type == 'commute':
        covP = torch.cov(XP.T) + reg * torch.eye(XP.shape[1])
        covQ = torch.cov(XQ.T) + reg * torch.eye(XQ.shape[1])

        DP, VP = torch.linalg.eig(covP)
        DP = torch.diag(DP ** (1 / 2))
        covP_sqrt = (VP @ DP @ VP.T).real

        DQ, VQ = torch.linalg.eig(covQ)
        DQ = torch.diag(DQ ** (1 / 2))
        covQ_sqrt = (VQ @ DQ @ VQ.T).real

        cov_diff = torch.linalg.norm(covP_sqrt - covQ_sqrt, ord='fro')
    elif cov_type == 'diag':
        stdP = torch.std(XP, dim=0)
        stdQ = torch.std(XQ, dim=0)

        cov_diff = torch.linalg.norm(stdP - stdQ) ** 2

    return torch.sqrt(mean_diff + cov_diff)


class WassersteinDistance(torch.nn.Module):
    r"""Wasserstein loss using the Primal Kantorovich formulation.
    Gradients are computed using the Envelope Theorem.

    Parameters
    ----------
    reg_e : float, optional (default=0.0)
        Entropic regularization penalty.
        If reg_e > 0, calculates the Sinkhorn Loss.
    n_iter_sinkhorn : int, optional (default=20)
        Maximum number of sinkhorn iterations. Only used for reg_e > 0.
    debias : bool, optional (default=False)
        whether or not compute the debiased sinkhorn loss.
        Only used when reg_e > 0.
    """
    def __init__(self, reg_e=0.0, n_iter_sinkhorn=20, debias=False):
        super(WassersteinDistance, self).__init__()
        self.reg_e = reg_e
        self.n_iter_sinkhorn = n_iter_sinkhorn
        self.debias = debias

    def forward(self, XP, XQ):
        r"""Computes the Wasserstien loss between samples XP ~ P and XQ ~ Q,

        Parameters
        ----------
        XP : tensor
            Tensor of shape (n, d) containing i.i.d samples from distribution P
        XQ : tensor
            Tensor of shape (m, d) containing i.i.d samples from distribution Q
        """
        uP = unif(XP.shape[0], device=XP.device)
        uQ = unif(XQ.shape[0], device=XQ.device)

        if self.debias and self.reg_e > 0.0:
            bias = 0.0

            CPP = torch.cdist(XP, XP, p=2) ** 2
            with torch.no_grad():
                plan_PP = log_sinkhorn(uP, uP, CPP / CPP.detach().max(),
                                       reg_e=self.reg_e,
                                       n_iter_max=self.n_iter_sinkhorn)
            bias += torch.sum(CPP * plan_PP)

            CQQ = torch.cdist(XQ, XQ, p=2) ** 2
            with torch.no_grad():
                plan_QQ = log_sinkhorn(uQ, uQ, CQQ / CQQ.detach().max(),
                                       reg_e=self.reg_e,
                                       n_iter_max=self.n_iter_sinkhorn)
            bias += torch.sum(CQQ * plan_QQ)

            CPQ = torch.cdist(XP, XQ, p=2) ** 2
            with torch.no_grad():
                plan_PQ = log_sinkhorn(uP, uQ, CPQ / CPQ.detach().max(),
                                       reg_e=self.reg_e,
                                       n_iter_max=self.n_iter_sinkhorn)
            loss_val = torch.sum(CPQ * plan_PQ)
            loss_val = loss_val - 0.5 * bias
        else:
            C = torch.cdist(XP, XQ, p=2) ** 2
            with torch.no_grad():
                if self.reg_e > 0.0:
                    π = log_sinkhorn(uP, uQ, C / C.detach().max(),
                                     reg_e=self.reg_e,
                                     n_iter_max=self.n_iter_sinkhorn)
                else:
                    π = emd(uP, uQ, C)
            loss_val = torch.sum(C * π)
        return loss_val


class SlicedWassersteinDistance(torch.nn.Module):
    r"""Sliced Wasserstein Distance"""
    def __init__(self, n_projections=20, use_max=False, p=2, q=1):
        self.use_max = use_max
        self.n_projections = n_projections
        self.p = p
        self.q = q
        super(SlicedWassersteinDistance, self).__init__()

    def forward(self, XP, XQ):
        return ot.sliced.sliced_wasserstein_distance(
            X_s=XP,
            X_t=XQ,
            n_projections=self.n_projections,
            p=self.p
        ) ** self.q


class BuresWassersteinDistance(torch.nn.Module):
    r"""Bures Wasserstein distance between distributions P and Q.
    Given samples XP and XQ from P and Q, computes the mean vectors
    :math:`\mu^{(P)}` and :math:`\mu^{(Q)}` and covariances
    :math:`\Sigma^{(P)}` and :math:`\Sigma^{(Q)}` and calculates
    the Bures-Wasserstein metric,

    .. math::
        W_{2}(P, Q) = \sqrt{\lVert \mu^{(P)} - \mu^{(Q)} \rVert_{2}^{2} +
        Tr(\Sigma^{(P)}) + Tr(\Sigma^{(Q)}) - 2Tr(M)}

    for :math:`M=((\Sigma^{(P)})^{1/2}\Sigma^{(Q)}(\Sigma^{(P)})^{1/2})^{1/2}`.

    Parameters
    ----------
    cov_type : str, optional (default='full')
        Either 'full', 'diag' or 'commute'. Determines how the Bures metric is
        computed.
    reg : float, optional (default=1e-6)
        Regularization for the covariance matrices.
    """
    def __init__(self, cov_type='full', reg=1e-6):
        if cov_type.lower() in ['full', 'diag', 'commute']:
            self.cov_type = cov_type.lower()
        else:
            self.cov_type = 'diag'
        self.reg = reg
        super(BuresWassersteinDistance, self).__init__()

    def forward(self, XP, XQ, index=None):
        return bures_wasserstein_metric(XP, XQ,
                                        cov_type=self.cov_type,
                                        reg=self.reg)


class JointWassersteinDistance(torch.nn.Module):
    r"""Wasserstein Metric between joint distributions of labels and features,
    using the Primal Kantorovich formulation. Gradients are computed using
    the Envelope Theorem.

    Parameters
    ----------
    reg_e : float, optional (default=0.0)
        Entropic regularization penalty.
    reg_m : float, optional (default=0.0)
        Unbalanced regularization penalty.
    label_weight : float, optional (default=None)
        Weight for the label metric. It is described
        as beta in the main paper. If None is given,
        uses beta as the maximum pairwise distance
        between samples of P and Q.
    n_iter_sinkhorn : int (default=20)
        Maximum number of sinkhorn iterations.
        Only used for reg_e > 0.
    label_metric : function
        Function that receives 2 matrices of labels
        of shape (n, nc) and (m, nc) and returns a
        matrix (n, m) of distances between labels.
    """
    def __init__(self,
                 reg_e=0.0,
                 reg_m=0.0,
                 label_weight=None,
                 label_metric=None,
                 n_iter_sinkhorn=20):
        super(JointWassersteinDistance, self).__init__()
        self.reg_e = reg_e
        self.reg_m = reg_m
        self.label_weight = label_weight
        self.n_iter_sinkhorn = n_iter_sinkhorn

        if label_metric is not None:
            self.label_metric = label_metric
        else:
            self.label_metric = lambda YP, YQ: torch.cdist(YP, YQ, p=2) ** 2

    def forward(self, XQ, YQ, XP, YP, index=None):
        r"""Computes the Wasserstien loss between samples (XP, YP) ~ P and
        (XQ, YQ) ~ Q,

        Parameters
        ----------
        XP : tensor
            Tensor of shape (n, d) containing features from samples from P
        YP : tensor
            Tensor of shape (n, nc) containing labels from samples from P.
            If None is given, ignores labels and computes the standard
            Wasserstein distance.
        XQ : tensor
            Tensor of shape (m, d) containing features from samples from Q
        YQ : tensor
            Tensor of shape (n, nc) containing labels from samples from Q.
            If None is given, ignores labels and computes the standard
            Wasserstein distance.
        """
        a = unif(XP.shape[0], device=XP.device)
        b = unif(XQ.shape[0], device=XQ.device)
        CX = torch.cdist(XP, XQ, p=2) ** 2

        if YP is not None and YQ is not None:
            CY = self.label_metric(YP, YQ)
        else:
            CY = torch.zeros_like(CX)

        if CY.detach().max() == 0.0:
            label_weight = 0.0
        else:
            if self.label_weight is None:
                label_weight = (CX.detach().max() / CY.detach().max())
            else:
                label_weight = self.label_weight

        C = CX + label_weight * CY
        with torch.no_grad():
            if self.reg_e > 0.0:
                if self.reg_m > 0.0:
                    n_iter = self.n_iter_sinkhorn
                    plan = log_unbalanced_sinkhorn(a, b, C / C.max(),
                                                   reg_e=self.reg_e,
                                                   reg_m=self.reg_m,
                                                   n_iter_max=n_iter)
                else:
                    plan = log_sinkhorn(a, b, C / C.max(),
                                        reg_e=self.reg_e,
                                        n_iter_max=self.n_iter_sinkhorn)
            else:
                plan = emd(a, b, C)
        return torch.sum(C * plan)


class ClassConditionalBuresWassersteinDistance(torch.nn.Module):
    r"""Class-conditional version of BuresWassersteinDistance. Implements,

    .. math::
        WC_{2}(P, Q) = \frac{1}{n_{c}}\sum_{c=1}^{n_{c}}W_{2}(P_{c},Q_{c})

    where :math:`P_{c}` correspond to the class-conditional distribution
    :math:`P_{c} = P(X|Y=c)`.

    Parameters
    ----------
    cov_type : str, optional (default='full')
        Either 'full', 'diag' or 'commute'. Determines how the Bures metric is
        computed.
    reg : float, optional (default=1e-6)
        Regularization for the covariance matrices.
    """
    def __init__(self, cov_type='full', reg=1e-6):
        if cov_type.lower() in ['full', 'diag', 'commute']:
            self.cov_type = cov_type.lower()
        else:
            self.cov_type = 'diag'
        self.reg = reg
        super(ClassConditionalBuresWassersteinDistance, self).__init__()

    def forward(self, XP, YP, XQ, YQ):
        r"""Computes the Class-Condtional Bures-Wasserstein loss between
        samples (XP, YP) ~ P and (XQ, YQ) ~ Q,

        Parameters
        ----------
        XP : tensor
            Tensor of shape (n, d) containing features from samples from P
        YP : tensor
            Tensor of shape (n, nc) containing labels from samples from P.
            If None is given, ignores labels and computes the standard
            Wasserstein distance.
        XQ : tensor
            Tensor of shape (m, d) containing features from samples from Q
        YQ : tensor
            Tensor of shape (n, nc) containing labels from samples from Q.
            If None is given, ignores labels and computes the standard
            Wasserstein distance.
        """
        if YP is not None and YQ is not None:
            loss = 0.0
            for c in range(YP.shape[1]):
                ind_Pc = torch.where(YP.argmax(dim=1) == c)[0]
                ind_Qc = torch.where(YQ.argmax(dim=1) == c)[0]

                XPc = XP[ind_Pc]
                XQc = XQ[ind_Qc]

                loss += bures_wasserstein_metric(XPc, XQc,
                                                 cov_type=self.cov_type,
                                                 reg=self.reg)
        else:
            loss = self.__bures_wasserstein(XP, XQ,
                                            cov_type=self.cov_type,
                                            reg=self.reg)

        return loss


class ClassConditionalWassersteinDistance(torch.nn.Module):
    r"""Wasserstein loss between joint distributions of
    labels and features, using the Primal Kantorovich formulation.
    Gradients are computed using the Envelope Theorem."""
    def __init__(self, reg_e=0.0, reg_m=0.0, n_iter_sinkhorn=20, p=2, q=2):
        r"""Creates the loss object.

        Parameters
        ----------
        reg_e : float, optional (default=0)
            Entropic regularization penalty. If 0, uses the EMD insetad.
        reg_m : float, optional (default=0)
            Marginal OT plan relaxation. If greater than 0, computes
            unbalanced OT.
        n_iter_sinkhorn : int, optional (default=20)
            Maximum number of sinkhorn iterations. Only used for reg_e > 0.
        """
        super(ClassConditionalWassersteinDistance, self).__init__()
        self.reg_e = reg_e
        self.reg_m = reg_m
        self.p = p
        self.q = q
        self.n_iter_sinkhorn = n_iter_sinkhorn

    def forward(self, XQ, YQ, XP, YP):
        r"""Computes the Class-Conditional Wasserstien loss between samples
        XP ~ P and XQ ~ Q,

        Parameters
        ----------
        XP : torch Tensor
            Tensor of shape (n, d) containing i.i.d features
            from distribution P
        YP: torch Tensor
            Tensor of shape (n, nc) containing i.i.d labels
            from distribution P
        XQ: torch Tensor
            Tensor of shape (m, d) containing i.i.d samples
            from distribution Q
        YQ: torch Tensor
            Tensor of shape (n, nc) containing i.i.d labels
            from distribution Q
        """
        if YP is not None and YQ is not None:
            yP, yQ = YP.argmax(dim=1), YQ.argmax(dim=1)

            loss = 0.0
            for c in yP.unique():
                indP, indQ = torch.where(yP == c)[0], torch.where(yQ == c)[0]
                if len(indP) > 0.0 and len(indQ) > 0.0:
                    _XP, _XQ = XP[indP], XQ[indQ]

                    a = unif(_XP.shape[0], device=_XP.device)
                    b = unif(_XQ.shape[0], device=_XQ.device)
                    C = torch.cdist(_XP, _XQ, p=self.p) ** self.q

                    with torch.no_grad():
                        if self.reg_e > 0.0:
                            π = log_sinkhorn(a, b, C / C.detach().max(),
                                             reg_e=self.reg_e,
                                             n_iter_max=self.n_iter_sinkhorn)
                        else:
                            π = emd(a, b, C)
                    loss += torch.sum(C * π)
        else:
            a = unif(XP.shape[0], device=XP.device)
            b = unif(XQ.shape[0], device=XQ.device)
            C = torch.cdist(XP, XQ, p=self.p) ** self.q
            with torch.no_grad():
                if self.reg_e > 0.0:
                    π = log_sinkhorn(a, b, C / C.detach().max(),
                                     reg_e=self.reg_e,
                                     n_iter_max=self.n_iter_sinkhorn)
                else:
                    π = emd(a, b, C)
            loss = torch.sum(C * π)
        return loss
