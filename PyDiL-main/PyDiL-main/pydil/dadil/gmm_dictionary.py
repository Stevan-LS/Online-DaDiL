import ot
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pydil.ot_utils.pot_utils import proj_simplex
from pydil.prob_utils.gauss import gmm_density_1d


class GaussianMixtureDictionary(torch.nn.Module):
    r"""Labeled dictionary class. This class implements
    a dictionary whose atoms are empirical distributions
    with labels. This serves for domain adaptation, or
    for multi-task learning, depending on whether domains
    are labeled or not.

    Parameters
    ----------
    XP : list of tensors
        Initialization for atoms features. Should be a list or tensor
        of shape (K, n, d), where K is the number of components,
        n is the number of samples, and d is the dimension of
        the feature space. If None, samples features are initialized
        randomly.
    YP : list of tensors
        Initialization for atoms labels. Should be a list or tensor
        of shape (K, n, nc), where K is the number of components,
        n is the number of samples and nc is the number of classes
        in the problem. If None, samples classes are initialized
        randomly.
    A : tensor
        Matrix of shape (N, K), where each row represents the
        barycentric coordinates of a given dataset. If None,
        it is initialized randomly depending on weight_initialization.
    n_samples : int, optional (default=1024)
        Only used if (XP, YP) are None. Number of samples in the
        atoms support.
    n_dim : int, optional (default=None)
        Only used if (XP, YP) are None. Dimensionality of feature
        space
    n_classes : int, optional (default=None)
        Only used if (XP, YP) are None. Number of classes in the
        classification problem.
    n_distributions : int, optional (default=None)
        Only used if A is None. Number of distributions in the
        dictionary learning problem.
    loss_fn : function, optional (default=None)
        Loss function between distributions. If None, uses
        `pydil.ipms.ot_ipms.JointWassersteinDistance`.
    learning_rate_features : float, optional (default=1e-1)
        Learning rate for atoms features.
    learning_rate_labels : float, optional (default=None)
        Learning rate for atoms labels. If None, uses the same
        value as features.
    learning_rate_weights : float, optional (default=None)
        Learning rate for weights. If None, uses the same
        value as features.
    reg_e : float, optional (default=0.0)
        Penalty for entropic regularization.
    n_iter_barycenter : int, optional (default=10)
        Number of iterations in the Wasserstein
        barycenter algorithm
    n_iter_sinkhorn : int, optional (default=20)
        Number of iterations in the Sinkhorn
        algorithm. Only used if reg_e > 0.0.
    n_iter_emd : int, optional (default=1000000)
        Number of iterations in the EMD algorithm.
        Only used if reg_e = 0.0
    domain_names : list, optional (default=None)
        Names for domains in the DiL problem. If
        not given, uses "Domain ℓ" as a generic
        name.
    grad_labels : bool, optional (default=True)
        Whether to optimize w.r.t. atoms labels.
    optimizer_name : str, optional (default='adam')
        Name of optimizer. Either 'adam' or 'sgd'
    balanced_sampling : bool, optional (default=True)
        Whether to sample balanced batches from
        atoms.
    sampling with replacement : bool, optional (default=True)
        Whether to sample from atoms with replacement.
    barycenter_tol : float, (default=1e-9)
        Stopping criteria for Wasserstein barycenter.
    barycenter_beta : float, optional (default=None)
        Importance of label distance in Wasserstein barycenter
        algorithm. If None is given, uses maximum distance
        between samples features.
    tensor_dtype : torch dtype, optional (default=torch.float32)
        Dtype of tensors.
    track_atoms : bool, optional (default=False)
        If True, saves atoms at each iteration. NOTE: depending
        on the problem (e.g., n_samples and n_dim), setting
        this parameter to True can have a large memory consumption.
    """
    def __init__(self,
                 n_distributions,
                 n_dim,
                 n_classes,
                 n_components=2,
                 weight_initialization='random',
                 learning_rate_atoms=1e-1,
                 learning_rate_weights=None,
                 domain_names=None,
                 optimizer_name='sgd',
                 tensor_dtype=torch.float32,
                 track_atoms=False,
                 schedule=False,
                 weight_dil=0.0,
                 weight_gmm=1.0,
                 momentum=0.9):
        super(GaussianMixtureDictionary, self).__init__()

        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.weight_initialization = weight_initialization
        self.n_distributions = n_distributions
        self.tensor_dtype = tensor_dtype
        self.learning_rate_atoms = learning_rate_atoms
        self.schedule = schedule
        self.momentum = momentum
        self.weight_dil = weight_dil
        self.weight_gmm = weight_gmm

        if learning_rate_weights is None:
            self.learning_rate_weights = self.learning_rate_atoms
        else:
            self.learning_rate_weights = learning_rate_weights

        if domain_names is None:
            self.domain_names = [
                "Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

        self.optimizer_name = optimizer_name
        self.track_atoms = track_atoms

        self.__initialize_atoms()
        self.__initialize_weights()

        self.history = {
            'loss': [],
            'loss_dil': [],
            'loss_gmm': [],
            'weights': [],
            'atoms': [],
            'loss_per_dataset': {name: [] for name in self.domain_names}
        }

    def __initialize_atoms(self):
        MP_data = torch.randn(self.n_components, self.n_classes, self.n_dim)
        SP_data = torch.ones(self.n_components, self.n_classes, self.n_dim)
        BP_data = torch.stack([
            torch.ones(self.n_classes) / self.n_classes
            for _ in range(self.n_components)
        ])

        self.means = torch.nn.parameter.Parameter(data=MP_data,
                                                  requires_grad=True)
        self.stds = torch.nn.parameter.Parameter(data=SP_data,
                                                 requires_grad=True)
        self.class_weights = torch.nn.parameter.Parameter(data=BP_data,
                                                          requires_grad=True)

    def __initialize_weights(self, A=None):
        if A is None:
            if self.n_distributions is None:
                raise ValueError(("If 'A' is not given you"
                                  " should specify 'n_distributions'"))
            if self.weight_initialization == 'random':
                a_data = torch.rand(self.n_distributions,
                                    self.n_components,
                                    requires_grad=True).to(self.tensor_dtype)
            else:
                a_data = torch.ones(self.n_distributions,
                                    self.n_components,
                                    requires_grad=True).to(self.tensor_dtype)
            with torch.no_grad():
                a_data = proj_simplex(a_data.T).T
        else:
            a_data = A
        self.A = torch.nn.parameter.Parameter(data=a_data, requires_grad=True)

    def configure_optimizers(self):
        r"""Returns optimizers for dictionary variables."""
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD([
                {'params': self.means,
                 'lr': self.learning_rate_atoms,
                 'momentum': self.momentum},
                {'params': self.stds,
                 'lr': self.learning_rate_atoms,
                 'momentum': self.momentum},
                {'params': self.class_weights,
                 'lr': self.learning_rate_atoms,
                 'momentum': self.momentum},
                {'params': self.A,
                 'lr': self.learning_rate_weights,
                 'momentum': self.momentum}
            ])
        else:
            return torch.optim.Adam([
                {'params': self.means, 'lr': self.learning_rate_atoms},
                {'params': self.stds, 'lr': self.learning_rate_atoms},
                {'params': self.class_weights, 'lr': self.learning_rate_atoms},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])

    def get_atoms(self):
        r"""Gets a list containing atom parameters."""
        return (self.means.detach(),
                self.stds.detach(),
                self.class_weights.detach())

    def get_weights(self):
        r"""Gets barycentric coordinates."""
        with torch.no_grad():
            if (self.A.sum(dim=1) == 1).all():
                return self.A.data.cpu()
            else:
                return proj_simplex(self.A.data.cpu().T).T

    def estimate_log_likelihood(self, x, weights):
        MB = torch.einsum('k,kcd->cd', weights, self.means)
        SB = torch.einsum('k,kcd->cd', weights, self.stds)
        BB = torch.einsum('k,kc->c', weights, self.class_weights)

        log_probs = []
        for mk, sk, bk in zip(MB, SB, BB):
            log_prob = - ((((x - mk) ** 2) / (2 * (sk ** 2))).sum(dim=1) +
                          (x.shape[1] / 2) * np.log(2 * np.pi) +
                          torch.log(torch.prod(sk))) + torch.log(bk)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)
        log_likelihood = - torch.logsumexp(log_probs, dim=0).mean()

        return log_likelihood

    def estimate_sw_distance(self, x, means, covs, weights,
                             n_proj=20, n_bins=30):
        projections = torch.randn(n_proj, x.shape[1])
        projections / torch.linalg.norm(projections, dim=1)[:, None]

        # x is (n, d), _x is (n, nproj)
        _x = x @ projections.T
        _means = means @ projections.T
        _covs = torch.einsum('ni,kij,nj->nk', projections, covs, projections)

        loss = 0.0
        for proj_idx in range(n_proj):
            hist, edges = torch.histogram(_x[:, proj_idx],
                                          bins=n_bins,
                                          density=True)
            t = .5 * (edges[1:] + edges[:-1])
            _mproj = _means[:, proj_idx]
            _sproj = torch.sqrt(_covs[proj_idx, :])

            gmm_f = gmm_density_1d(t, _mproj, _sproj, weights=weights,
                                   log=False)
            with torch.no_grad():
                hist = hist / hist.sum()
                gmm_f = gmm_f / gmm_f.sum()
            loss += ot.lp.wasserstein_1d(u_values=t,
                                         v_values=t,
                                         u_weights=hist,
                                         v_weights=gmm_f,
                                         p=2) / n_proj
        return loss

    def fit_dataloader(self, BQ, MQ, SQ, target_loader, n_iter_max=100):
        self.optimizer = self.configure_optimizers()
        if self.schedule:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

        for it in range(n_iter_max):
            # Calculates the loss
            MB = torch.einsum('lk,kcd->lcd', self.A, self.means)
            SB = torch.einsum('lk,kcd->lcd', self.A, self.stds)
            BB = torch.einsum('lk,kc->lc', self.A, self.class_weights)

            # DiL on GMM parameters
            loss_sources = (((MB[:-1] - MQ) ** 2).mean(dim=[0, 1]).sum() +
                            ((SB[:-1] - SQ) ** 2).mean(dim=[0, 1]).sum() +
                            ((BB[:-1] - BQ) ** 2).mean(dim=0).sum())

            # Maximum Likelihood on target
            loss_target = 0.0
            for xt in target_loader:
                loss_target += self.estimate_log_likelihood(xt,
                                                            weights=self.A[-1])
            loss_target /= len(target_loader)

            # Total Loss
            loss = loss_sources + loss_target

            # Optimizer Step
            loss.backward()
            self.optimizer.step()

            # Projection
            with torch.no_grad():
                self.A.data = proj_simplex(self.A.data.T).T
                _w = self.class_weights.data.cpu()
                self.class_weights.data = proj_simplex(_w.T).T

            # Saves history info
            m = self.means.detach().data.cpu().clone()
            s = self.stds.detach().data.cpu().clone()
            w = self.class_weights.detach().data.cpu().clone()
            self.history['atoms'].append({'means': m,
                                          'stds': s,
                                          'class_weights': w})
            self.history['weights'].append(proj_simplex(self.A.data.T).T)
            self.history['loss'].append(loss.item())
            self.history['loss_sources'].append(loss_sources.item())
            self.history['loss_target'].append(loss_target.item())
            print('It {}/{}, Loss: {}'.format(it, n_iter_max, loss.item()))
            if self.schedule:
                self.scheduler.step(loss.item())
        self.fitted = True

    def fit_log_likelihood(self, BQ, MQ, SQ, X, n_iter_max=100):
        self.optimizer = self.configure_optimizers()
        if self.schedule:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

        for it in range(n_iter_max):
            self.optimizer.zero_grad()

            # Calculates the loss
            MB = torch.einsum('lk,kcd->lcd', self.A[:-1], self.means)
            SB = torch.einsum('lk,kcd->lcd', self.A[:-1], self.stds)
            BB = torch.einsum('lk,kc->lc', self.A[:-1], self.class_weights)

            # DiL on GMM parameters
            loss_sources = (((MB - MQ) ** 2).mean(dim=[0, 1]).sum() +
                            ((SB - SQ) ** 2).mean(dim=[0, 1]).sum() +
                            ((BB - BQ) ** 2).mean(dim=0).sum())

            # Maximum Likelihood on target
            nll = 0
            for ak, Xk in zip(self.A, X):
                nll += self.estimate_log_likelihood(Xk, weights=ak)

            # Total Loss
            loss = self.weight_dil * loss_sources + self.weight_gmm * nll

            # Optimizer Step
            loss.backward()
            self.optimizer.step()

            # Projection
            with torch.no_grad():
                self.A.data = proj_simplex(self.A.data.T).T
                _w = self.class_weights.data.cpu()
                self.class_weights.data = proj_simplex(_w.T).T
                self.stds[self.stds < 0.0] = 0.1

            # Saves history info
            m = self.means.detach().data.cpu().clone()
            s = self.stds.detach().data.cpu().clone()
            w = self.class_weights.detach().data.cpu().clone()
            A = proj_simplex(self.A.detach().cpu().data.T).T.clone()
            self.history['atoms'].append({'means': m,
                                          'stds': s,
                                          'class_weights': w})
            self.history['weights'].append(A)
            self.history['loss'].append(loss.item())
            self.history['loss_dil'].append(loss_sources.item())
            self.history['loss_gmm'].append(nll.item())
            print('It {}/{}, Loss: {}'.format(it, n_iter_max, loss.item()))
            if self.schedule:
                self.scheduler.step(loss.item())
        self.fitted = True

    def fit_swgmm(self, BQ, MQ, SQ, X, n_iter_max=100):
        self.optimizer = self.configure_optimizers()
        if self.schedule:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

        for it in range(n_iter_max):
            self.optimizer.zero_grad()

            # Calculates the loss
            MB = torch.einsum('lk,kcd->lcd', self.A, self.means)
            SB = torch.einsum('lk,kcd->lcd', self.A, self.stds)
            BB = torch.einsum('lk,kc->lc', self.A, self.class_weights)

            # DiL on GMM parameters
            loss_sources = (((MB[:-1] - MQ) ** 2).mean(dim=[0, 1]).sum() +
                            ((SB[:-1] - SQ) ** 2).mean(dim=[0, 1]).sum() +
                            ((BB[:-1] - BQ) ** 2).mean(dim=0).sum())

            # Maximum Likelihood on target
            loss_gmm = 0.0
            for Xl, MBl, SBl, BBl in zip(X, MB, SB, BB):
                _SBl = torch.stack([torch.diag(SBlk) for SBlk in SBl])
                loss_gmm += self.estimate_sw_distance(Xl,
                                                      MBl,
                                                      _SBl,
                                                      BBl,
                                                      n_proj=20,
                                                      n_bins=30)

            # Total Loss
            loss = loss_sources + loss_gmm

            # Optimizer Step
            loss.backward()
            self.optimizer.step()

            # Projection
            with torch.no_grad():
                self.A.data = proj_simplex(self.A.data.T).T
                _w = self.class_weights.data.cpu()
                self.class_weights.data = proj_simplex(_w.T).T
                self.stds[self.stds < 0.0] = 0.1

            # Saves history info
            m = self.means.detach().data.cpu().clone()
            s = self.stds.detach().data.cpu().clone()
            w = self.class_weights.detach().data.cpu().clone()
            A = proj_simplex(self.A.detach().cpu().data.T).T.clone()
            self.history['atoms'].append({'means': m,
                                          'stds': s,
                                          'class_weights': w})
            self.history['weights'].append(A)
            self.history['loss'].append(loss.item())
            self.history['loss_dil'].append(loss_sources.item())
            self.history['loss_gmm'].append(loss_gmm.item())
            print('It {}/{}, Loss: {}'.format(it, n_iter_max, loss.item()))
            if self.schedule:
                self.scheduler.step(loss.item())
        self.fitted = True

    def fit_gmm(self, BQ, MQ, SQ, n_iter_max=100):
        self.optimizer = self.configure_optimizers()
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        for it in range(n_iter_max):
            self.optimizer.zero_grad()

            # Calculates the loss
            MB = torch.einsum('lk,kcd->lcd', self.A, self.means)
            SB = torch.einsum('lk,kcd->lcd', self.A, self.stds)
            BB = torch.einsum('lk,kc->lc', self.A, self.class_weights)

            # DiL on GMM parameters
            loss = (((MB - MQ) ** 2).mean(dim=[0, 1]).sum() +
                    ((SB - SQ) ** 2).mean(dim=[0, 1]).sum() +
                    ((BB - BQ) ** 2).mean(dim=0).sum())

            # Optimizer Step
            loss.backward()
            self.optimizer.step()

            # Projection
            with torch.no_grad():
                self.A.data = proj_simplex(self.A.data.T).T
                _w = self.class_weights.data.cpu()
                self.class_weights.data = proj_simplex(_w.T).T
                self.stds[self.stds < 0.0] = 0.1

            # Saves history info
            m = self.means.detach().data.cpu().clone()
            s = self.stds.detach().data.cpu().clone()
            w = self.class_weights.detach().data.cpu().clone()
            A = proj_simplex(self.A.detach().cpu().data.T).T.clone()
            self.history['atoms'].append({'means': m,
                                          'stds': s,
                                          'class_weights': w})
            self.history['weights'].append(A)
            self.history['loss'].append(loss.item())
            print('It {}/{}, Loss: {}'.format(it, n_iter_max, loss.item()))
            self.scheduler.step(loss.item())
        self.fitted = True

    def map_to_target(self, x, y, domain):
        weights_d = self.A[domain, :]
        weights_T = self.A[-1, :]

        means_d = torch.einsum('k,kcd->cd', weights_d, self.means)
        stds_d = torch.einsum('k,kcd->cd', weights_d, self.stds)

        means_T = torch.einsum('k,kcd->cd', weights_T, self.means)
        stds_T = torch.einsum('k,kcd->cd', weights_T, self.stds)

        mapped_x, mapped_y = [], []
        for c in range(self.n_classes):
            m_dc, s_dc = means_d[c], stds_d[c]
            m_Tc, s_Tc = means_T[c], stds_T[c]

            A = torch.diag(s_Tc / s_dc)
            b = m_Tc - m_dc @ A

            ind = torch.where(y == c)[0]
            mapped_x.append(x[ind] @ A + b)
            mapped_y.extend(y[ind])

        return torch.cat(mapped_x, dim=0), torch.Tensor(mapped_y)
