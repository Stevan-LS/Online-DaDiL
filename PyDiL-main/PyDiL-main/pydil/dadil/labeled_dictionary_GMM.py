import torch
import numpy as np

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pydil.ot_utils.pot_utils import proj_simplex
from pydil.ipms.ot_ipms import JointWassersteinDistance
from pydil.ot_utils.barycenters import wasserstein_barycenter
import sys
sys.path.append('../../')
import Online_GMM


class LabeledDictionaryGMM(torch.nn.Module):
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
                 loss_fn=None,
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
                 balanced_sampling=True,
                 sampling_with_replacement=True,
                 barycenter_tol=1e-9,
                 barycenter_beta=None,
                 tensor_dtype=torch.float32,
                 track_atoms=False,
                 schedule_lr=True,
                 GMM_components=13,
                 GMM_dim_reduction=3):
        super(LabeledDictionaryGMM, self).__init__()

        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.weight_initialization = weight_initialization
        self.n_distributions = n_distributions
        self.tensor_dtype = tensor_dtype
        self.learning_rate_features = learning_rate_features

        if loss_fn is None:
            self.loss_fn = JointWassersteinDistance()
        else:
            self.loss_fn = loss_fn

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
        self.balanced_sampling = balanced_sampling
        self.sampling_with_replacement = sampling_with_replacement
        self.barycenter_tol = barycenter_tol
        self.barycenter_beta = barycenter_beta
        self.track_atoms = track_atoms
        self.var_tracker = [torch.zeros(n_samples)
                            for _ in range(self.n_components)]
        self.schedule_lr = schedule_lr

        self.__initialize_atoms_features(XP)
        self.__initialize_atoms_labels(YP)
        self.__initialize_weights(A)

        self.history = {
            'loss': [],
            'weights': [],
            'atoms_features': [],
            'atoms_labels': [],
            'loss_per_dataset': {name: [] for name in self.domain_names}
        }

        self.OGMM = None
        self.GMM_components = GMM_components
        self.GMM_dim_reduction = GMM_dim_reduction

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

    def configure_optimizers(self, regularization=False):
        r"""Returns optimizers for dictionary variables."""
        if regularization:
            if self.grad_labels:
                return torch.optim.Adam([
                    {'params': self.XP, 'lr': self.learning_rate_features, 'weight_decay': self.learning_rate_features/10},
                    {'params': self.YP, 'lr': self.learning_rate_labels, 'weight_decay': self.learning_rate_labels/10},
                    {'params': self.A, 'lr': self.learning_rate_weights}
                ])
            else:
                return torch.optim.Adam([
                    {'params': self.XP, 'lr': self.learning_rate_features, 'weight_decay': self.learning_rate_features/10},
                    {'params': self.A, 'lr': self.learning_rate_weights}
                ])
        else:
            if self.grad_labels:
                return torch.optim.Adam([
                    {'params': self.XP, 'lr': self.learning_rate_features},
                    {'params': self.YP, 'lr': self.learning_rate_labels},
                    {'params': self.A, 'lr': self.learning_rate_weights}
                ])
            else:
                return torch.optim.Adam([
                    {'params': self.XP, 'lr': self.learning_rate_features},
                    {'params': self.A, 'lr': self.learning_rate_weights}
                ])

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

    def sample_from_atoms(self, n=None, detach=False):
        """Samples (with replacement) $n$ samples from atoms support.

        Parameters
        ----------
        n : int, optional (default=None)
            Number of samples (with replacement) acquired from the atoms
            support. If None, gets all samples from the atoms supports.
        detach : bool, optional (default=False).
            If True, detaches tensors so that gradients are not calculated.
        """
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for tracker, XPk, YPk in zip(self.var_tracker, self.XP, self.YP):
            # If balanced sampling, needs to select sampler_per_class
            # from each class
            if self.balanced_sampling:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    # Gets indices from current class
                    ind = torch.where(yPk == yu)[0]
                    # Randomly permutes labels
                    perm = torch.randperm(len(ind))
                    ind = ind[perm]
                    if samples_per_class is None:
                        # If n was not given, samples all samples
                        # from the said class
                        sampled_indices.append(ind[:])
                    else:
                        # Samples "samples_per_class" from given class
                        sampled_indices.append(ind[:samples_per_class])
                # Concatenates all indices
                sampled_indices = torch.cat(sampled_indices, dim=0)
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples),
                                                   size=n)

            # Adds counter of sampling
            tracker[sampled_indices] += 1

            # Creates batch arrays
            features_k, labels_k = XPk[sampled_indices], YPk[sampled_indices]

            if self.grad_labels:
                labels_k = labels_k.softmax(dim=-1)

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()

            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels

    def batch_generator(self, batch_size):
        r"""Creates a generator of batches from
        atoms without replacement.

        Parameters
        ----------
        batch_size : int
            Number of samples in mini-batches.
        """
        n_batches = self.n_samples // batch_size
        n_classes_per_batch = batch_size // self.n_classes

        for i in range(n_batches + 1):
            batch_indices = []
            for YPk in self.YP:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                atom_batch_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    indices = np.where(yPk == yu)[0]
                    atom_batch_indices.append(
                        indices[n_classes_per_batch * i:
                                n_classes_per_batch * (i + 1)]
                    )
                atom_batch_indices = np.concatenate(atom_batch_indices)
                np.random.shuffle(atom_batch_indices)
                batch_indices.append(atom_batch_indices)
            yield batch_indices

    def fit_without_replacement(self,
                                datasets,
                                n_iter_max=100):
        r"""Minimizes DaDiL objective function w.r.t. atoms
        features and weights, by sampling from atoms without
        replacement of samples.

        Parameters
        ----------
        datasets : list of measures
            List of measure objects, which implement sampling from
            datasets.
        n_iter_max : int, optional (default=100)
            Number of epoch in DaDiL's optimization
        """
        self.optimizer = self.configure_optimizers()
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        batch_size = datasets[0].batch_size

        for it in range(n_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(datasets))}
            idx_gen = self.batch_generator(batch_size=batch_size)

            __n_batches = 0
            for batch_indices in idx_gen:
                self.optimizer.zero_grad()

                XP = [XPk[ind_k] for XPk, ind_k in zip(self.XP, batch_indices)]
                YP = [YPk[ind_k] for YPk, ind_k in zip(self.YP, batch_indices)]

                loss = 0
                for ℓ, (Qℓ, αℓ) in enumerate(zip(datasets, self.A)):
                    XQℓ, YQℓ = Qℓ.sample()
                    XBℓ, YBℓ = wasserstein_barycenter(
                        XP, YP=YP, XB=None, YB=None,
                        weights=αℓ, n_samples=batch_size,
                        reg_e=self.reg_e, label_weight=self.barycenter_beta,
                        n_iter_max=self.n_iter_barycenter,
                        n_iter_sinkhorn=self.n_iter_sinkhorn,
                        n_iter_emd=self.n_iter_emd, tol=self.barycenter_tol,
                        verbose=False, inner_verbose=False, log=False,
                        propagate_labels=True, penalize_labels=True)

                    loss_ℓ = self.loss_fn(XQ=XQℓ, YQ=YQℓ, XP=XBℓ, YP=YBℓ)
                    loss += loss_ℓ
                    loss_val = loss_ℓ.detach().cpu().item()
                    avg_it_loss_per_dataset[self.domain_names[ℓ]] += loss_val

                loss.backward()
                self.optimizer.step()

                # Projects the weights into the simplex
                with torch.no_grad():
                    self.A.data = proj_simplex(self.A.data.T).T

                avg_it_loss += loss.item()
                __n_batches += 1

            for ℓ in len(datasets):
                avg_it_loss_per_dataset[self.domain_names[ℓ]] /= __n_batches
            avg_it_loss /= __n_batches

            # Saves history info
            _XP, _YP = self.get_atoms()
            self.history['atoms_features'].append(_XP)
            self.history['atoms_labels'].append(_YP)
            self.history['weights'].append(proj_simplex(self.A.data.T).T)
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    avg_it_loss_per_dataset[self.domain_names[ℓ]]
                )
            print('It {}/{}, Loss: {}'.format(it, n_iter_max, avg_it_loss))
            if self.schedule_lr:
                self.scheduler.step(avg_it_loss)
        self.fitted = True

    def fit(self,
            datasets,
            n_iter_max=100,
            batches_per_it=10,
            verbose=True,
            regularization=False):
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
        self.optimizer = self.configure_optimizers(regularization=regularization)
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        batch_size = datasets[0].batch_size
        for it in range(n_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(datasets))}
            if verbose:
                pbar = tqdm(range(batches_per_it))
            else:
                pbar = range(batches_per_it)
            for _ in pbar:
                self.optimizer.zero_grad()

                loss = 0
                for ℓ, (Qℓ, αℓ) in enumerate(zip(datasets, self.A)):
                    # Sample minibatch from dataset
                    XQℓ, YQℓ = Qℓ.sample(batch_size)

                    # Sample minibatch from atoms
                    XP, YP = self.sample_from_atoms(n=batch_size)

                    # Calculates Wasserstein barycenter
                    XBℓ, YBℓ = wasserstein_barycenter(
                        XP, YP=YP, XB=None, YB=None,
                        weights=αℓ, n_samples=batch_size,
                        reg_e=self.reg_e, label_weight=None,
                        n_iter_max=self.n_iter_barycenter,
                        n_iter_sinkhorn=self.n_iter_sinkhorn,
                        n_iter_emd=self.n_iter_emd, tol=self.barycenter_tol,
                        verbose=False, inner_verbose=False, log=False,
                        propagate_labels=True, penalize_labels=True)

                    # Calculates Loss
                    loss_ℓ = self.loss_fn(XQ=XQℓ, YQ=YQℓ, XP=XBℓ, YP=YBℓ)

                    # Accumulates loss
                    loss += loss_ℓ
                    loss_val = loss_ℓ.detach().cpu().item() / batches_per_it
                    avg_it_loss_per_dataset[self.domain_names[ℓ]] += loss_val

                loss.backward()
                self.optimizer.step()

                # Projects the weights into the simplex
                with torch.no_grad():
                    self.A.data = proj_simplex(self.A.data.T).T

                avg_it_loss += loss.item() / batches_per_it
            # Saves history info
            _XP, _YP = self.get_atoms()
            self.history['atoms_features'].append(_XP)
            self.history['atoms_labels'].append(_YP)
            self.history['weights'].append(proj_simplex(self.A.data.T).T)
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    avg_it_loss_per_dataset[self.domain_names[ℓ]]
                )
            if verbose:
                print('It {}/{}, Loss: {}'.format(it, n_iter_max, avg_it_loss))
            if self.schedule_lr:
                self.scheduler.step(avg_it_loss)
        self.fitted = True
    
    def fit_target_sample(self,
            target_sample,
            batches_per_it=10,
            batch_size=128,
            verbose=True,
            regularization=False):
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
        if self.OGMM == None:
            self.OGMM = Online_GMM(
                n_components=self.GMM_components,
                lr=0.1,
                n_features=self.GMM_dim_reduction,
                data_range=torch.mean(torch.max(torch.concat(list(self.XP), axis = 0), axis=0).values - 
                                      torch.min(torch.concat(list(self.XP), axis = 0), axis=0).values).item(),
                batch_size=batch_size
            )
        self.OGMM.fit_sample(target_sample, dimension_reduction=True)

        self.optimizer = self.configure_optimizers(regularization=regularization)
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        for it in range(target_sample.shape[0]):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {
                self.domain_names[0]: 0 }
            if verbose:
                pbar = tqdm(range(batches_per_it))
            else:
                pbar = range(batches_per_it)
            for _ in pbar:
                self.optimizer.zero_grad()

                loss = 0
                for ℓ, (Qℓ, αℓ) in enumerate(zip(datasets, self.A)):
                    # Sample minibatch from dataset
                    XQℓ, YQℓ = Qℓ.sample(batch_size)

                    # Sample minibatch from atoms
                    XP, YP = self.sample_from_atoms(n=batch_size)

                    # Calculates Wasserstein barycenter
                    XBℓ, YBℓ = wasserstein_barycenter(
                        XP, YP=YP, XB=None, YB=None,
                        weights=αℓ, n_samples=batch_size,
                        reg_e=self.reg_e, label_weight=None,
                        n_iter_max=self.n_iter_barycenter,
                        n_iter_sinkhorn=self.n_iter_sinkhorn,
                        n_iter_emd=self.n_iter_emd, tol=self.barycenter_tol,
                        verbose=False, inner_verbose=False, log=False,
                        propagate_labels=True, penalize_labels=True)

                    # Calculates Loss
                    loss_ℓ = self.loss_fn(XQ=XQℓ, YQ=YQℓ, XP=XBℓ, YP=YBℓ)

                    # Accumulates loss
                    loss += loss_ℓ
                    loss_val = loss_ℓ.detach().cpu().item() / batches_per_it
                    avg_it_loss_per_dataset[self.domain_names[ℓ]] += loss_val

                loss.backward()
                self.optimizer.step()

                # Projects the weights into the simplex
                with torch.no_grad():
                    self.A.data = proj_simplex(self.A.data.T).T

                avg_it_loss += loss.item() / batches_per_it
            # Saves history info
            _XP, _YP = self.get_atoms()
            self.history['atoms_features'].append(_XP)
            self.history['atoms_labels'].append(_YP)
            self.history['weights'].append(proj_simplex(self.A.data.T).T)
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    avg_it_loss_per_dataset[self.domain_names[ℓ]]
                )
            if verbose:
                print('It {}/{}, Loss: {}'.format(it, n_iter_max, avg_it_loss))
            if self.schedule_lr:
                self.scheduler.step(avg_it_loss)
        self.fitted = True

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
