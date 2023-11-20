import torch


class MMDLoss(torch.nn.Module):
    r"""Maximum Mean Discrepancy distance of [Gretton et al., 2012]

    Parameters
    ----------
    kernel : string, optional (default='linear')
        Name of the kernel. Either 'linear' or 'rbf'.
    gamma : float, optional (default=None)
        Parameter of the RBF kernel. Only used if kernel='linear'.
    """
    def __init__(self, kernel='linear', gamma=None, pow_max=2):
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.pow_max = pow_max

    def forward(self, XP, XQ):
        r"""Computes the MMD loss between samples XP ~ P and XQ ~ Q,

        Parameters
        ----------
        XP : tensor
            Tensor of shape (n, d) containing i.i.d samples from distribution P
        XQ : tensor
            Tensor of shape (m, d) containing i.i.d samples from distribution Q
        """
        if self.kernel.lower() == 'linear':
            return torch.linalg.norm(XP.mean(dim=0) - XQ.mean(dim=0))
        elif self.kernel.lower() == 'moment-matching':
            loss = 0.0
            for k in range(1, self.pow_max + 1):
                loss += torch.linalg.norm(
                    (XP ** k).mean(dim=0) - (XQ ** k).mean(dim=0))
            return loss
        elif self.kernel.lower() == 'rbf':
            if self.gamma is None:
                gamma = 1 / XP.shape[1]
            else:
                gamma = self.gamma
            Kpp = torch.exp(- gamma * torch.cdist(XP, XP, p=2))
            Kpq = torch.exp(- gamma * torch.cdist(XP, XQ, p=2))
            Kqq = torch.exp(- gamma * torch.cdist(XQ, XQ, p=2))
            return Kpp.mean() + Kqq.mean() - 2 * Kpq.mean()
        else:
            raise ValueError(f"Kernel {self.kernel} not implemented.")


class ClassConditionalMMD(torch.nn.Module):
    r"""MMD between class conditional distributions.

    Parameters
    ----------
    kernel : string, optional (default='linear')
        Name of the kernel. Either 'linear' or 'rbf'.
    gamma : float, optional (default=None)
        Parameter of the RBF kernel. Only used if kernel='linear'.
    """
    def __init__(self, kernel='linear', gamma=None):
        super(ClassConditionalMMD, self).__init__()
        self.kernel = kernel
        self.gamma = gamma

    def forward(self, XQ, YQ, XP, YP):
        r"""Computes the Class-Condtional MMD loss between samples
        (XP, YP) ~ P and (XQ, YQ) ~ Q,

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
            yQ = YQ.argmax(dim=1)
            yP = YP.argmax(dim=1)

            loss = 0.0
            for c in range(YP.shape[1]):
                ind_P = torch.where(yP == c)[0]
                ind_Q = torch.where(yQ == c)[0]

                if len(ind_P) > 0 and len(ind_Q) > 0:
                    XPc = XP[ind_P]
                    XQc = XQ[ind_Q]

                    if self.kernel.lower() == 'linear':
                        loss += torch.linalg.norm(
                            XPc.mean(dim=0) - XQc.mean(dim=0))
                    elif self.kernel.lower() == 'rbf':
                        if self.gamma is None:
                            gamma = 1 / XP.shape[1]
                        else:
                            gamma = self.gamma
                        Kpp = torch.exp(- gamma * torch.cdist(XPc, XPc, p=2))
                        Kpq = torch.exp(- gamma * torch.cdist(XPc, XQc, p=2))
                        Kqq = torch.exp(- gamma * torch.cdist(XQc, XQc, p=2))
                        loss += Kpp.mean() + Kqq.mean() - 2 * Kpq.mean()
                    else:
                        raise ValueError(f"Kernel {self.kernel}"
                                         " not implemented.")
        else:
            loss = torch.linalg.norm(XP.mean(dim=0) - XQ.mean(dim=0))

        return loss
