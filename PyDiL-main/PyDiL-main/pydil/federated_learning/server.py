import torch
import numpy as np
from pydil.ot_utils.barycenters import wasserstein_barycenter


class DaDiLServer:
    def __init__(self,
                 n_samples,
                 n_dim,
                 n_classes,
                 n_components,
                 balanced_sampling=False,
                 aggregation='random',
                 track_atoms=False,
                 track_atom_versions=False):
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.balanced_sampling = balanced_sampling
        self.aggregation = aggregation
        self.track_atoms = track_atoms
        self.track_atom_versions = track_atom_versions

        if self.track_atoms:
            print("WARNING: Tracking atom throughout optimization. ",
                  "This choice may be memory intensive. Use with caution.")

        if self.track_atom_versions:
            print("WARNING: Tracking atom versions throughout optimization. ",
                  "This choice may be memory intensive. Use with caution.")

        # initializes atoms
        self.XP, self.YP = self.initialize_params()

        # History
        self.history = {'atoms': [], 'versions': []}

    def initialize_params(self):
        """Randomly initializes dictionary samples"""
        # Forcing same number of samples per class
        samples_per_class = self.n_samples // self.n_classes
        if self.n_samples % self.n_classes != 0:
            self.n_samples = self.n_classes * samples_per_class

        # Initializing Atom Features
        XP = [
            torch.randn(self.n_samples, self.n_dim).float()
            for _ in range(self.n_components)
        ]

        # Initializing Atom Labels
        YP = []
        for _ in range(self.n_components):
            ypk = torch.cat(
                [torch.tensor([c] * samples_per_class)
                    for c in range(self.n_classes)]
            ).long()
            YPk = torch.nn.functional.one_hot(
                ypk, num_classes=self.n_classes).float()
            YP.append(YPk)
        return XP, YP

    def sample_from_atoms(self, n=None):
        """Samples (with replacement) $n$ samples from atoms support.

        Parameters
        ----------
        n : int, optional (default=None)
            Number of samples (with replacement) acquired from the atoms
            support. If None, gets all samples from the atoms supports.
        """
        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        batch_indices = []
        # Sampling
        for YPk in self.YP:
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
                    if samples_per_class is None:
                        # If n was not given, samples all samples
                        # from the said class
                        sampled_indices.append(ind[:])
                    else:
                        # Samples "samples_per_class" from given class
                        sampled_indices.append(
                            torch.from_numpy(
                                np.random.choice(ind, size=samples_per_class)
                            ).long()
                        )
                # Concatenates all indices
                sampled_indices = torch.cat(sampled_indices, dim=0)
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples),
                                                   size=n)
            batch_indices.append(sampled_indices)

        return batch_indices

    def avg_aggregation(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms via linear interpolation."""
        XPv, YPv = torch.stack(XP_versions), torch.stack(YP_versions)
        new_XP = [
            XPv.mean(dim=0)[k, ...]
            for k in range(self.n_components)]
        new_YP = [
            YPv.mean(dim=0)[k, ...]
            for k in range(self.n_components)]
        return new_XP, new_YP

    def barycentric_aggregation(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms via Wasserstein barycenter"""
        new_XP, new_YP = [], []

        for k in range(self.n_components):
            _X, _Y = wasserstein_barycenter(
                XP=[XPv[k, ...] for XPv in XP_versions],
                YP=[YPv[k, ...] for YPv in YP_versions],
                XB=None,
                YB=None,
                weights=None,
                tol=1e-9,
                n_iter_max=10,
                n_samples=len(XP_versions[0][0]),
                propagate_labels=True,
                penalize_labels=True
            )
            new_XP.append(_X)
            new_YP.append(_Y)

        return new_XP, new_YP

    def random_aggregation(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms by randomly choosing
        one version."""
        available_versions = np.arange(len(XP_versions))
        selected_client = np.random.choice(available_versions)
        new_XP, new_YP = [], []
        for k in range(self.n_components):
            new_XP.append(XP_versions[selected_client][k])
            new_YP.append(YP_versions[selected_client][k])
        return new_XP, new_YP

    def aggregate(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms by different rules."""
        if len(XP_versions) > 1:
            """Multiple versions of atoms exist. We need to aggregate
            them."""
            if self.aggregation == 'avg':
                return self.avg_aggregation(XP_versions, YP_versions)
            elif self.aggregation == 'wbary':
                return self.barycentric_aggregation(XP_versions, YP_versions)
            elif self.aggregation == 'random':
                return self.random_aggregation(XP_versions, YP_versions)
        else:
            """Only one version of atoms exist. Return as is."""
            return XP_versions[0], YP_versions[0]

    def federated_fit(self,
                      clients,
                      spc=10,
                      n_iter=1,
                      n_client_it=1,
                      C=1,
                      verbose=False):
        """Fits a dictionary via Federated Learning.

        Parameters
        ----------
        clients : list of Clients
            List of client objects.
        spc : int
            Number of samples per class sampled at each iteration.
        n_iter : int
            Number of Dictionary Learning iterations
        n_client_it : int
            Number of iterations taken by each client.
        C : float
            Proportion of clients sampled at each iteration
        verbose : bool
            If True, prints information about iterations
        """
        batch_size = spc * self.n_classes
        client_list = np.arange(len(clients))
        n_sampled_clients = max([
            np.round(len(clients) * C).astype(int), 1
        ])
        for it in range(n_iter):
            if verbose:
                message = f"Round {it}"
                print(message)
                print('-' * len(message))

            batch_indices = self.sample_from_atoms(n=batch_size)

            XP_versions = []
            YP_versions = []

            selected_clients = np.random.choice(client_list,
                                                size=n_sampled_clients,
                                                replace=False)

            for selected_client in selected_clients:
                if verbose:
                    message = f'Client {selected_client}'
                    print(message)
                    print('-' * len(message))

                client = clients[selected_client]

                local_XP = [XPk[ind_k].clone()
                            for ind_k, XPk in zip(batch_indices, self.XP)]
                local_YP = [YPk[ind_k].clone()
                            for ind_k, YPk in zip(batch_indices, self.YP)]

                _XP, _YP = client.update_joint(local_XP,
                                               local_YP,
                                               n_iter=n_client_it,
                                               verbose=verbose)
                XP_versions.append(torch.stack(_XP))
                YP_versions.append(torch.stack(_YP))

                if verbose:
                    print('\n')

            if self.track_atom_versions:
                self.history['versions'].append(
                    [[XPv.clone() for XPv in XP_versions],
                     [YPv.clone() for YPv in YP_versions]]
                )

            new_XP, new_YP = self.aggregate(XP_versions, YP_versions)
            for k in range(self.n_components):
                self.XP[k][batch_indices[k]] = new_XP[k]
                self.YP[k][batch_indices[k]] = new_YP[k]

            if self.track_atoms:
                self.history['atoms'].append(
                    [[XPk.clone() for XPk in self.XP],
                     [YPk.clone() for YPk in self.YP]]
                )

    def federated_full_fit(self,
                           clients,
                           batch_size,
                           n_iter=1,
                           n_client_it=1,
                           C=1,
                           verbose=False):
        client_list = np.arange(len(clients))
        n_sampled_clients = max([
            np.round(len(clients) * C).astype(int), 1
        ])
        for it in range(n_iter):
            if verbose:
                message = f"Round {it}"
                print(message)
                print('-' * len(message))

            XP_versions = []
            YP_versions = []

            selected_clients = np.random.choice(client_list,
                                                size=n_sampled_clients,
                                                replace=False)

            for selected_client in selected_clients:
                if verbose:
                    message = f'Client {selected_client}'
                    print(message)
                    print('-' * len(message))

                client = clients[selected_client]

                # Creates local copies of atom variables
                local_XP = [XPk.clone() for XPk in self.XP]
                local_YP = [YPk.clone() for YPk in self.YP]

                _XP, _YP = client.update_joint(
                    local_XP, local_YP, batch_size=batch_size,
                    n_iter=n_client_it, verbose=verbose,
                    balanced_sampling_atoms=self.balanced_sampling)

                XP_versions.append(torch.stack(_XP))
                YP_versions.append(torch.stack(_YP))

                if verbose:
                    print('\n')

            if self.track_atom_versions:
                self.history['versions'].append(
                    [[XPv.clone() for XPv in XP_versions],
                     [YPv.clone() for YPv in YP_versions]]
                )

            # Aggregates multiple versions
            new_XP, new_YP = self.aggregate(XP_versions, YP_versions)
            for k in range(self.n_components):
                self.XP[k].data = new_XP[k].data.clone()
                self.YP[k].data = new_YP[k].data.clone()

            if self.track_atoms:
                self.history['atoms'].append(
                    [[XPk.clone() for XPk in self.XP],
                     [YPk.clone() for YPk in self.YP]]
                )
