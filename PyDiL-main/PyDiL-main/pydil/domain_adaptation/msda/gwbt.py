import torch
from pydil.linalg_utils.matrices import sqrtm
from pydil.ot_utils.barycenters import bures_wasserstein_barycenter


class GWBT(torch.nn.Module):
    r"""Wasserstein Barycenter Transport under Gaussian
    assumption. Assumes that data follows a Gaussian distribution
    for calculating Monge Mappings between distributions.

    Parameters
    ----------
    n_domains : int
        Number of source domains
    n_dim : int
        Number of dimensions in the data
    requires_grad : bool, optional (default=False)
        If True, linear maps are differentiable.
    cov_type : str, optional (defualt='full')
        Either 'full', 'diag' or 'commute', corresponding
        to the assumptions on the covariance matrices
        of domains.
    dtype : torch.dtype, optional (default=torch.float32)
        Data type for tensors.
    verbose : bool, optional (default=False)
        If True, plots information about barycenters
    n_iter_barycenter : int, optional (default=10)
        Number of fixed-point iterations in the barycenter
        algorithm.
    reg : float, optional (default=0.0)
        Regularization for covariance matrices.
    """
    def __init__(self,
                 n_domains,
                 n_dim,
                 requires_grad=False,
                 cov_type='full',
                 dtype=torch.float32,
                 verbose=False,
                 tol=1e-9,
                 n_iter_barycenter=10,
                 reg=0.0):
        super(GWBT, self).__init__()

        self.n_domains = n_domains
        self.n_dim = n_dim
        self.cov_type = cov_type.lower()
        self.dtype = dtype
        self.verbose = verbose
        self.tol = tol
        self.n_iter_barycenter = n_iter_barycenter
        self.reg = reg

        A_sb = torch.randn(n_domains, n_dim, n_dim, dtype=dtype)
        self.A_sb = torch.nn.Parameter(data=A_sb, requires_grad=requires_grad)
        A_bt = torch.randn(n_dim, n_dim, dtype=dtype)
        self.A_bt = torch.nn.Parameter(data=A_bt, requires_grad=requires_grad)
        b_sb = torch.randn(n_domains, n_dim, dtype=dtype)
        self.b_sb = torch.nn.Parameter(data=b_sb, requires_grad=requires_grad)
        b_bt = torch.randn(n_dim, dtype=dtype)
        self.b_bt = torch.nn.Parameter(data=b_bt, requires_grad=requires_grad)

    def __get_transform(self, Σa, Σb):
        sqrtm_Σa_pos, sqrtm_Σa_neg = sqrtm(Σa, return_inv=True)
        M = sqrtm(sqrtm_Σa_pos @ Σb @ sqrtm_Σa_pos)

        return sqrtm_Σa_neg @ M @ sqrtm_Σa_neg

    def fit(self, Xs, Xt):
        μs, σs, Σs = [], [], []
        for Xsk in Xs:
            μs.append(Xsk.mean(dim=0)[None, ...])
            σs.append(Xsk.std(dim=0)[None, ...])
            Σs.append(
                (((1 - self.reg) * torch.cov(Xsk.T) +
                  self.reg * torch.eye(self.n_dim))[None, ...])
            )
        μs = torch.cat(μs, dim=0)
        σs = torch.cat(σs, dim=0)
        Σs = torch.cat(Σs, dim=0)
        μt = Xt.mean(dim=0)
        σt = Xt.std(dim=0)
        Σt = ((1 - self.reg) * torch.cov(Xt.T)
              + self.reg * torch.eye(self.n_dim))

        w = (torch.ones(self.n_domains) / self.n_domains).to(self.dtype)
        if self.cov_type == 'full' or self.cov_type == 'commute':
            μB, ΣB = bures_wasserstein_barycenter(
                μs, Σs, w, num_iter_max=self.n_iter_barycenter,
                cov_type=self.cov_type, tol=self.tol,
                verbose=self.verbose, extra_ret=False)
        elif self.cov_type == 'diag':
            μB, σB = bures_wasserstein_barycenter(μs, σs, w,
                                                  cov_type=self.cov_type)

        # Mapping from each source to the barycenter
        for k, (μsk, σsk, Σsk) in enumerate(zip(μs, σs, Σs)):
            if self.cov_type == 'full':
                self.A_sb[k] = self.__get_transform(Σsk, ΣB)
            elif self.cov_type == 'diag':
                self.A_sb[k] = torch.diag(σB / σsk)
            self.A_sb[k].data[self.A_sb[k].isinf()] = 0.0
            self.b_sb[k] = μB - self.A_sb[k] @ μsk

        # Mapping from the barycenter to the target
        if self.cov_type == 'full':
            self.A_bt.data = self.__get_transform(ΣB, Σt)
        elif self.cov_type == 'diag':
            self.A_bt.data = torch.diag(σt / σB)
        self.A_bt.data[self.A_bt.isinf()] = 0.0
        self.b_bt.data = μt - self.A_bt @ μB

    def transform(self, Xs):
        return torch.cat([
            (Xsk @ Ak + bk) @ self.A_bt + self.b_bt
            for Ak, bk, Xsk in zip(self.A_sb, self.b_sb, Xs)
        ], dim=0)
