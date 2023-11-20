import torch
import numpy as np
from pydil.ot_utils.pot_utils import proj_simplex
from pydil.ot_utils.barycenters import wasserstein_barycenter


class PersistentClient(torch.nn.Module):
    def __init__(self,
                 features,
                 n_components,
                 criterion,
                 n_classes,
                 n_samples,
                 optimizer_name='sgd',
                 momentum=0.9,
                 lr=1e-1,
                 lr_weights=None,
                 labels=None,
                 batches_per_it=10,
                 balanced_sampling=False,
                 grad_labels=True):
        super(PersistentClient, self).__init__()
        self.features = features
        self.labels = labels
        self.n_classes = n_classes
        self.n_dim = features.shape[1]
        self.n_samples = n_samples
        self.ind = np.arange(len(features))
        self.n_components = n_components
        self.criterion = criterion
        self.lr = lr
        if lr_weights is None:
            self.lr_weights = lr
        else:
            self.lr_weights = lr_weights
        self.momentum = momentum
        self.optimizer_name = optimizer_name
        self.batches_per_it = batches_per_it
        self.balanced_sampling = balanced_sampling
        self.grad_labels = grad_labels
        self.n_samples = len(features)

        # Initializes weights
        self.weights = torch.ones(n_components) / n_components
        self.weights.requires_grad = True

        # Local copy of dictoinary
        self.XP, self.YP = self.initialize_params()

        # Initializes optimizer
        self.optimizer = self.get_optimizer()

        # History
        self.history = {
            'weights': [self.weights.data.clone()],
            'loss': []
        }

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

        for XPk, YPk in zip(XP, YP):
            XPk.requires_grad = True
            YPk.requires_grad = True
        return XP, YP

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': self.XP, 'lr': self.lr, 'momentum': self.momentum},
                {'params': self.YP, 'lr': self.lr, 'momentum': self.momentum},
                {'params': self.weights, 'lr': self.lr_weights,
                 'momentum': self.momentum}
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': self.XP, 'lr': self.lr},
                {'params': self.YP, 'lr': self.lr},
                {'params': self.weights, 'lr': self.lr_weights}
            ])
        return optimizer

    def sample(self, n=None):
        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        if self.balanced_sampling and self.labels is not None:
            # Gets categorical labels
            yQ = self.labels.detach().argmax(dim=1)
            # Initializes list of sampled indices
            sampled_indices = []
            # Loops over each class
            for yu in yQ.unique():
                # Gets indices from current class
                ind = torch.where(yQ == yu)[0]
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
        batch_features = self.features[sampled_indices]
        if self.labels is not None:
            batch_labels = self.labels[sampled_indices]
        else:
            batch_labels = None

        return batch_features, batch_labels

    def update_joint(self, XP, YP, batch_size, n_iter, verbose,
                     balanced_sampling_atoms):
        # Clones global versions into local ones
        with torch.no_grad():
            for k in range(self.n_components):
                self.XP[k].data = XP[k].data
                self.YP[k].data = YP[k].data

        for it in range(n_iter):
            # Creating mini-batch indices
            yP = [YPk.argmax(dim=1) for YPk in YP]
            n_batches = len(XP[0]) // batch_size
            spc = batch_size // self.n_classes
            if balanced_sampling_atoms:
                batch_indices = []
                for b in range(n_batches):
                    batch_idx = []
                    for k in range(self.n_components):
                        batch_idx.append(torch.cat([
                            torch.where(yP[k] == c)[0][
                                b * spc: (b + 1) * spc
                            ]
                            for c in range(self.n_classes)
                        ], dim=0))
                batch_indices.append(batch_idx)
            else:
                indices = np.arange(len(XP[0]))
                np.random.shuffle(indices)
                batch_indices = [
                    indices[i * batch_size: (i + 1) * batch_size]
                    for i in range(n_batches)
                ]

            # Iteration loss accumulator
            it_loss = 0
            for batch_idx in batch_indices:
                self.optimizer.zero_grad()

                # Randomly samples features and labels
                XQ, YQ = self.sample(n=batch_size)

                # Gets current minibatch from atoms
                if balanced_sampling_atoms:
                    minibatch_XP = [
                        XPk[batch_idx_k]
                        for XPk, batch_idx_k in zip(self.XP, batch_idx)
                    ]
                    minibatch_YP = [
                        YPk[batch_idx_k].softmax(dim=1)
                        for YPk, batch_idx_k in zip(self.YP, batch_idx)
                    ]
                else:
                    minibatch_XP = [XPk[batch_idx] for XPk in self.XP]
                    minibatch_YP = [
                        YPk[batch_idx].softmax(dim=1) for YPk in self.YP]

                # Computes a Wasserstein Barycenter
                XB, YB = wasserstein_barycenter(
                    XP=minibatch_XP, YP=minibatch_YP, XB=None, YB=None,
                    weights=self.weights, n_samples=batch_size,
                    reg_e=0.0, label_weight=None, tol=1e-9,
                    propagate_labels=True, penalize_labels=True
                )

                # Computes loss
                loss = self.criterion(XB, YB, XQ, YQ)

                # Backpropagation
                loss.backward()

                # Optimization step
                self.optimizer.step()

                # Projection back into the simplex
                with torch.no_grad():
                    self.weights.data = proj_simplex(self.weights.data)
                    self.history['weights'].append(self.weights.data.clone())

                # Accumulates on it_loss
                it_loss += loss.item() / self.batches_per_it
            self.history['loss'].append(it_loss)
            if verbose:
                print(f"Local Iter {it}, loss {it_loss}")
        return ([XPk.detach().data for XPk in self.XP],
                [YPk.detach().data for YPk in self.YP])
