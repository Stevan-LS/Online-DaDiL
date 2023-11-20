import torch
import warnings
import numpy as np


class UnsupervisedDatasetMeasure:
    r"""Unsupervised Dataset Measure class.
    Samples elements from datasets with replacement.
    Datasets are assumed unsupervised, i.e., no annotations
    are available for features.

    Parameters
    ----------
    features : np.array
        Numpy array of shape (n, d) containing features.
    transforms : list of functions
        pre-processing steps for data samples.
    batch_size : int, optional (default=64)
        Number of elements in batches.
    device : str
        Either 'cpu' or 'cuda', corresponding to the devie
        of returned batches.
    """
    def __init__(self, features, transforms=None, batch_size=64, device='cpu'):
        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn(("Trying to use gpu when no"
                           " device is available. Using CPU instead."))
            device = 'cpu'
        self.device = torch.device(device)

        self.features = features
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_dim = features.shape[1]
        self.ind = np.arange(len(features))

    def sample(self, n=None):
        r"""Samples $n$ points from the measure support.

        Parameters
        ----------
        n : int, optional (default=None)
            If given, samples n samples from the support.
            By default samples batch_size elements.
        """
        n = self.batch_size if n is None else n
        minibatch_ind = np.random.choice(self.ind, size=n)
        minibatch_features = self.features[minibatch_ind]

        if self.transforms is not None:
            minibatch_features = torch.cat([
                self.transforms(xi)[None, ...]for xi in minibatch_features],
                dim=0)
        elif type(minibatch_features) == np.ndarray:
            minibatch_features = torch.from_numpy(minibatch_features)
        return minibatch_features.to(self.device), None


class SupervisedDatasetMeasure:
    r"""Supervised Dataset Measure class.
    Samples elements from datasets with replacement.
    Datasets are assumed supervised, i.e., to each feature
    there corresponds a categorical annotation.

    Parameters
    ----------
    features : np.array
        Numpy array of shape (n, d) containing features.
    labels : np.array
        Numpy array of shape (n,) containing categorical labels.
    transforms : list of functions
        pre-processing steps for data samples.
    batch_size : int, optional (default=64)
        Number of elements in batches.
    device : str
        Either 'cpu' or 'cuda', corresponding to the devie
        of returned batches.
    """
    def __init__(self,
                 features,
                 labels,
                 transforms=None,
                 batch_size=64,
                 stratify=False,
                 device='cpu'):
        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn(("Trying to use gpu when no"
                           " device is available. Using CPU instead."))
            device = 'cpu'
        self.device = torch.device(device)
        self.labels = labels
        self.features = features
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_dim = features.shape[1]
        self.n_classes = len(np.unique(labels))
        self.ind = np.arange(len(features))
        self.stratify = stratify

        self.ind_per_class = [
            np.where(labels == yu)[0] for yu in np.unique(labels)
        ]

    def sample(self, n=None):
        r"""Samples $n$ points from the measure support.

        Parameters
        ----------
        n : int, optional (default=None)
            If given, samples n samples from the support.
            By default samples batch_size elements.
        """
        n = self.batch_size if n is None else n
        if self.stratify:
            samples_per_class = n // self.n_classes
            minibatch_ind = np.concatenate([
                np.random.choice(indices, size=samples_per_class)
                for indices in self.ind_per_class
            ])
        else:
            minibatch_ind = np.random.choice(self.ind, size=n)
        minibatch_labels = self.labels[minibatch_ind]
        minibatch_features = self.features[minibatch_ind]

        if self.transforms is not None:
            minibatch_features = torch.cat(
                [self.transforms(xi)[None, ...] for xi in minibatch_features],
                dim=0)
        elif type(minibatch_features) == np.ndarray:
            minibatch_features = torch.from_numpy(minibatch_features)

        if type(minibatch_labels) == np.ndarray:
            minibatch_labels = torch.nn.functional.one_hot(
                torch.from_numpy(minibatch_labels).long(),
                num_classes=self.n_classes).float()
        else:
            minibatch_labels = torch.nn.functional.one_hot(
                minibatch_labels.long(), num_classes=self.n_classes).float()

        return (minibatch_features.to(self.device),
                minibatch_labels.to(self.device))
