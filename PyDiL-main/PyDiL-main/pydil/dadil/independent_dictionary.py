import torch
import numpy as np

from tqdm.auto import tqdm

from pydil.ot_utils.pot_utils import emd
from pydil.ot_utils.pot_utils import proj_simplex
from pydil.ipms.ot_ipms import WassersteinDistance
from pydil.ot_utils.barycenters import wasserstein_barycenter


class IndependentLabeledDictionary(torch.nn.Module):
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    schedule_lr : bool, optional (default=True)
        If True, schedules learning rate of optimizer.
    """
    def __init__(self,
                 XP=None,
                 YP=None,
                 A=None,
                 n_samples=1024,
                 n_dim=None,
                 n_classes=None,
                 n_components=2,
                 weight_initialization='random',
                 n_distributions=None,
                 criterion_features=None,
                 criterion_labels=None,
                 learning_rate_features=1e-1,
                 learning_rate_labels=None,
                 learning_rate_weights=None,
                 reg_e=0.0,
                 n_iter_barycenter=10,
                 n_iter_sinkhorn=20,
                 n_iter_emd=1000000,
                 domain_names=None,
                 grad_labels=True,
                 optimizer_name='adam',
                 momentum=0.9,
                 balanced_sampling=True,
                 sampling_with_replacement=True,
                 barycenter_tol=1e-9,
                 barycenter_beta=None,
                 tensor_dtype=torch.float32,
                 track_atoms_features=False,
                 track_atoms_labels=False,
                 schedule_lr=True):
        super(IndependentLabeledDictionary, self).__init__()

        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.weight_initialization = weight_initialization
        self.n_distributions = n_distributions
        self.tensor_dtype = tensor_dtype
        self.learning_rate_features = learning_rate_features

        if criterion_features is None:
            self.criterion_features = WassersteinDistance()
        else:
            self.criterion_features = criterion_features

        if criterion_labels is None:
            self.criterion_labels = torch.nn.CrossEntropyLoss()
        else:
            self.criterion_labels = criterion_labels

        if learning_rate_labels is None:
            self.learning_rate_labels = learning_rate_features
        else:
            self.learning_rate_labels = learning_rate_labels

        if learning_rate_weights is None:
            self.learning_rate_weights = self.learning_rate_features
        else:
            self.learning_rate_weights = learning_rate_weights

        if domain_names is None:
            if n_distributions is None:
                raise ValueError(("If 'domain_names' is not given,"
                                  " 'n_distributions' must be provided."))
            self.domain_names = [
                "Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

        self.reg_e = reg_e
        self.n_iter_barycenter = n_iter_barycenter
        self.n_iter_sinkhorn = n_iter_sinkhorn
        self.n_iter_emd = n_iter_emd
        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.balanced_sampling = balanced_sampling
        self.sampling_with_replacement = sampling_with_replacement
        self.barycenter_tol = barycenter_tol
        self.barycenter_beta = barycenter_beta
        self.track_atoms_features = track_atoms_features
        self.track_atoms_labels = track_atoms_labels
        self.schedule_lr = schedule_lr

        self.__initialize_atoms_features(XP)
        self.__initialize_atoms_labels(YP)
        self.__initialize_weights(A)

        self.history = {
            'loss_features': [],
            'loss_labels': [],
            'weights': [],
            'atoms_features': [],
            'atoms_labels': [],
            'loss_per_dataset': {name: [] for name in self.domain_names}
        }

    def __initialize_atoms_features(self, XP=None):
        if XP is None:
            if self.n_dim is None:
                raise ValueError(("If 'XP' is not given,"
                                  " you should specify 'n_dim'."))
            XP_data = [
                torch.randn(self.n_samples, self.n_dim,
                            requires_grad=True).to(self.tensor_dtype)
                for _ in range(self.n_components)
            ]
            self.XP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=xp, requires_grad=True)
                 for xp in XP_data]
            )
        else:
            self.XP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=xp.to(self.tensor_dtype),
                                              requires_grad=True) for xp in XP]
            )
            self.n_dim = XP[0].shape[1]

    def __initialize_atoms_labels(self, YP=None):
        if YP is None:
            if self.n_classes is None:
                raise ValueError(("If 'YP' is not given,"
                                  " you should specify 'n_classes'"))
            samples_per_class = self.n_samples // self.n_classes
            if self.n_samples % self.n_classes != 0:
                self.n_samples = self.n_classes * samples_per_class
            YP_data = []
            for _ in range(self.n_components):
                ypk = torch.cat(
                    [torch.tensor([c] * samples_per_class)
                     for c in range(self.n_classes)]
                ).long()
                YPk = torch.nn.functional.one_hot(ypk,
                                                  num_classes=self.n_classes)
                YP_data.append(YPk)
            self.YP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype),
                                              requires_grad=self.grad_labels)
                 for yp in YP_data]
            )
        else:
            self.YP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype),
                                              requires_grad=self.grad_labels)
                 for yp in YP]
            )
            self.n_classes = YP[0].shape[1]

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
        r"""Sets variables optimizers."""
        if self.optimizer_name.lower() == 'adam':
            self.XP_optimizer = torch.optim.Adam(
                self.XP,
                lr=self.learning_rate_features)
            self.YP_optimizer = torch.optim.Adam(
                self.YP,
                lr=self.learning_rate_labels
            )
            self.A_optimizer = torch.optim.Adam(
                [self.A, ],
                lr=self.learning_rate_weights
            )
        else:
            self.XP_optimizer = torch.optim.SGD(
                self.XP,
                lr=self.learning_rate_features,
                momentum=self.momentum
            )
            self.YP_optimizer = torch.optim.SGD(
                self.YP,
                lr=self.learning_rate_labels,
                momentum=self.momentum
            )
            self.A_optimizer = torch.optim.SGD(
                [self.A, ],
                lr=self.learning_rate_weights,
                momentum=self.momentum
            )

    def get_atoms(self):
        r"""Gets a list containing atoms features and labels."""
        with torch.no_grad():
            XP = torch.stack([XPk.data.cpu() for XPk in self.XP])
            YP = torch.stack([YPk.data.cpu() for YPk in self.YP])
        return (XP, YP)

    def get_weights(self):
        r"""Gets barycentric coordinates."""
        with torch.no_grad():
            if (self.A.sum(dim=1) == 1).all():
                return self.A.data.cpu()
            else:
                return proj_simplex(self.A.data.cpu().T).T

    def fit(self,
            datasets,
            n_iter_max=100,
            batches_per_it=10,
            verbose=True):
        r"""Minimizes DaDiL's objective function by sampling
        mini-batches from the atoms support with replacement.

        Parameters
        ----------
        datasets : list of measures
            List of measure objects, which implement sampling from
            datasets.
        n_iter_max : int, optional (default=100)
            Number of epoch in DaDiL's optimization
        batches_per_it : int, optional (default=10)
            Number of batches drawn from the atoms per iteration.
        verbose : bool, optional (default=True)
            If True, prints progress of DaDiL's Optimization loop.
        """
        self.configure_optimizers()

        # Auxiliary variables
        batch_size = datasets[0].batch_size
        n_batches = self.n_samples // batch_size

        # Fit features
        print("Fitting Features")
        indices = np.arange(self.n_samples)
        for it in range(n_iter_max):
            np.random.shuffle(indices)
            batch_indices = [
                indices[i * batch_size: (i + 1) * batch_size]
                for i in range(n_batches)
            ]

            it_loss = 0.0
            for minibatch_idx in tqdm(batch_indices, total=n_batches):
                self.XP_optimizer.zero_grad()
                self.A_optimizer.zero_grad()
                minibatch_XP = [
                    XPk[minibatch_idx] for XPk in self.XP
                ]
                loss = 0.0
                for Qℓ, aℓ in zip(datasets, self.A):
                    minibatch_XQℓ, _ = Qℓ.sample()

                    XBℓ = wasserstein_barycenter(
                        XP=minibatch_XP,
                        YP=None,
                        XB=None,
                        YB=None,
                        weights=aℓ,
                        n_samples=batch_size,
                        reg_e=self.reg_e,
                        label_weight=None,
                        n_iter_max=self.n_iter_barycenter,
                        n_iter_sinkhorn=self.n_iter_sinkhorn,
                        n_iter_emd=self.n_iter_emd,
                        tol=self.barycenter_tol,
                        verbose=False,
                        inner_verbose=False,
                        propagate_labels=False,
                        penalize_labels=False,
                        log=False
                    )

                    loss += self.criterion_features(minibatch_XQℓ, XBℓ)
                loss.backward()
                self.XP_optimizer.step()
                self.A_optimizer.step()
                with torch.no_grad():
                    self.A.data = proj_simplex(self.A.data.T).T
                it_loss += loss.item() / n_batches
            self.history['weights'].append(
                proj_simplex(self.A.data.clone().T).T)
            self.history['loss_features'].append(it_loss)
            if self.track_atoms_features:
                _XP, _ = self.get_atoms()
                self.history['atoms_features'].append(_XP)

            print(f"Iter {it}, Loss {it_loss}")

        # Fit labels
        print("Fitting labels")
        indices = np.arange(self.n_samples)
        for it in range(n_iter_max):
            np.random.shuffle(indices)
            batch_indices = [
                indices[i * batch_size: (i + 1) * batch_size]
                for i in range(n_batches)
            ]

            it_loss = 0.0
            for minibatch_idx in tqdm(batch_indices, total=n_batches):
                self.YP_optimizer.zero_grad()
                minibatch_XP = [
                    XPk[minibatch_idx] for XPk in self.XP
                ]
                minibatch_YP = [
                    YPk[minibatch_idx].softmax(dim=1) for YPk in self.YP
                ]
                loss = 0.0
                for Qℓ, aℓ in zip(datasets, self.A):
                    minibatch_XQℓ, minibatch_YQℓ = Qℓ.sample()

                    if minibatch_YQℓ is None:
                        pass
                    else:
                        predicted_labels = 0
                        for aℓk, XPk, YPk in zip(aℓ, minibatch_XP,
                                                 minibatch_YP):
                            uPk = torch.ones(len(XPk)) / len(XPk)
                            uQl = (torch.ones(len(minibatch_XQl)) /
                                   len(minibatch_XQl))
                            C = torch.cdist(minibatch_XQl, XPk, p=2) ** 2
                            plan = emd(uQl, uPk, C)
                            predicted_labels += (aℓk.detach() *
                                                 len(uPk) *
                                                 torch.mm(plan, YPk))
                        loss += self.criterion_labels(
                            predicted_labels,
                            target=minibatch_YQℓ.argmax(dim=1).long())
                loss.backward()
                self.YP_optimizer.step()
                it_loss += loss.item() / n_batches
            self.history['loss_labels'].append(it_loss)
            if self.track_atoms_labels:
                _, _YP = self.get_atoms()
                self.history['atoms_labels'].append(_YP)

            print(f"Iter {it}, Loss {it_loss}")

    def transform(self,
                  datasets,
                  n_iter_max=100,
                  batches_per_it=10):
        r"""Represents a dataset in the simplex by regressing
        its weights with respect to fixed atoms.

        Parameters
        ----------
        datasets : list of measures
            List of measure objects, which implement sampling from
            datasets.
        n_iter_max : int, optional (default=100)
            Number of epoch in DaDiL's optimization
        batches_per_it : int, optional (default=10)
            Number of batches drawn from the atoms per iteration.
        """
        batch_size = datasets[0].batch_size
        embeddings = torch.randn(len(datasets),
                                 self.n_components,
                                 requires_grad=True,
                                 device=self.device)
        optimizer = torch.optim.Adam([embeddings], lr=self.lr)

        for step in range(n_iter_max):
            pbar = tqdm(range(batches_per_it))
            avg_it_loss = 0
            for _ in pbar:
                optimizer.zero_grad()
                loss = 0
                for Qℓ, αℓ in zip(datasets, embeddings):
                    XQℓ, YQℓ = Qℓ.sample()
                    XP, YP = self.sample_from_atoms(n=batch_size, detach=True)
                    XBℓ, YBℓ = wasserstein_barycenter(
                        XP, YP=YP, XB=None, YB=None,
                        weights=αℓ, n_samples=batch_size,
                        reg_e=self.reg_e, label_weight=None,
                        n_iter_max=self.n_iter_barycenter,
                        n_iter_sinkhorn=self.n_iter_sinkhorn,
                        n_iter_emd=self.n_iter_emd, tol=self.barycenter_tol,
                        verbose=False, inner_verbose=False, log=False,
                        propagate_labels=True, penalize_labels=True)
                    loss = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    embeddings = proj_simplex(
                        embeddings.data.cpu().T).T.to(self.device)

                avg_it_loss += loss.detach().item() / batches_per_it
                pbar.set_description(f"Loss: {loss.item()}")
            print('Step {:<4}/{:<4} loss: {:^15}'.format(step,
                                                         n_iter_max,
                                                         avg_it_loss))

        with torch.no_grad():
            return proj_simplex(
                self.embeddings.data.cpu().T).T.to(self.device)

    def reconstruct(self,
                    weights=None,
                    n_samples_atoms=None,
                    n_samples_barycenter=None):
        r"""Uses Wasserstein barycenters for reconstructing samples
        given weights

        Parameters
        ----------
        weights : tensor, optional (default=None)
            Tensor of shape (K,) with non-negative entries, which sum to 1.
            If None is given, uses self.A as weights.
        n_samples_atoms : int, optional (default=None)
            Number of samples sampled from each atom for the reconstruction.
            If None is given, uses all samples from atoms.
        n_samples_barycenter : int, optional (default=None)
            Number of samples in the barycenter's support. If None is given,
            uses the same number of samples from atoms.
        """
        if n_samples_atoms is None:
            n_samples_atoms = self.n_samples
        else:
            n_samples_atoms = n_samples_atoms

        if n_samples_barycenter is None:
            n_samples_barycenter = n_samples_atoms
        else:
            n_samples_barycenter
        XP, YP = self.sample_from_atoms(n=n_samples_atoms, detach=True)
        with torch.no_grad():
            if weights is None:
                # A = torch.nn.functional.softmax(self.weights, dim=1).detach()
                Q_rec = []
                for αℓ in self.A:
                    XB, YB = wasserstein_barycenter(
                        XP, YP=YP, XB=None, YB=None,
                        weights=αℓ, n_samples=n_samples_barycenter,
                        reg_e=self.reg_e, label_weight=None,
                        n_iter_max=self.n_iter_barycenter,
                        n_iter_sinkhorn=self.n_iter_sinkhorn,
                        n_iter_emd=self.n_iter_emd, tol=self.barycenter_tol,
                        verbose=False, inner_verbose=False, log=False,
                        propagate_labels=True, penalize_labels=True)
                    Q_rec.append([XB, YB])
            else:
                XB, YB = wasserstein_barycenter(
                    XP, YP=YP, XB=None, YB=None,
                    weights=αℓ, n_samples=n_samples_barycenter,
                    reg_e=self.reg_e, label_weight=None,
                    n_iter_max=self.n_iter_barycenter,
                    n_iter_sinkhorn=self.n_iter_sinkhorn,
                    n_iter_emd=self.n_iter_emd, tol=self.barycenter_tol,
                    verbose=False, inner_verbose=False, log=False,
                    propagate_labels=True, penalize_labels=True)
                Q_rec = [XB, YB]
        return Q_rec
