r"""Tabular Dataset Definitions.

In this module, we implement different versions of datasets
involving domain adaptation of "tabular" data, namely, datasets
in which data is in form of a table of n_samples rows and
n_features columns. This can either be purely tabular data, or
pre-extracted features from images, audio or text.
"""

import torch
import numpy as np


class FeaturesDataset(torch.utils.data.Dataset):
    r"""Standard Tabular dataset. Given a matrix of features X, of shape (n, d)
    and a matrix of labels Y, of shape (n, nc), this class implements a dataset
    where given idx, it returns (X[idx, :], Y[idx, :]).

    Parameters
    ----------
    features : tensor
        Tensor of shape (n, d) containing the feature vectors (in Rd) of
        samples.
    labels : tensor
        Tensor of shape (n, nc) containing the one-hot encoded labels of
        feature vectors.
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

        self.num_samples = len(features)
        self.num_features = self.features.shape[1]
        self.num_classes = self.labels.shape[1]

    def get_labels(self, one_hot=False):
        r"""Get labels corresponding to source domain(s)."""
        if one_hot:
            return self.labels.numpy()
        else:
            return self.labels.argmax(dim=1).numpy()

    def __len__(self):
        r"""Gets number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FeaturesSSDADataset(torch.utils.data.Dataset):
    r"""Single Source Domain Adaptation (SSDA) tabular dataset. This class
    receives three matrices, (Xs, Ys, Xt), of shapes (ns, d), (ns, nc) and
    (nt, d). Given a pair of indices, (src_idx, tgt_idx), returns three
    elements: (Xs[src_idx, :], Ys[src_idx, :], Xt[tgt_idx, :]). Optionally,
    one may provide Yt and Yt_hat, which are the labels and pseudo-labels
    of Xt.

    Parameters
    ----------
    source_features : tensor
        Tensor of shape (ns, d) containing the feature vectors (in Rd) of
        source domain samples.
    source_labels : tensor
        Tensor of shape (ns, nc) containing the one-hot encoded labels of
        source domain feature vectors.
    target_features : tensor
        Tensor of shape (n, d) containing the feature vectors (in Rd) of
        target domain samples.
    target_labels : tensor, optional (default=None)
        Tensor of shape (n, nc) containing the one-hot encoded labels of
        target domain feature vectors.
    target_pseudo_labels : tensor, optional (default=None)
        Tensor of shape (n, nc) containing the pseudo-labels of target
        domain feature vectors. These pseudo-labels correspond to
        probabilities of a sample belonging to a given class.
    """
    def __init__(self,
                 source_features,
                 source_labels,
                 target_features,
                 target_labels=None,
                 target_pseudo_labels=None):
        self.source_features = source_features
        self.source_labels = source_labels
        self.target_features = target_features
        self.target_labels = target_labels
        self.target_pseudo_labels = target_pseudo_labels

        self.num_domains = 1
        self.num_samples = np.min([len(source_features), len(target_features)])
        self.num_classes = self.source_labels.shape[1]
        self.num_features = self.source_features.shape[1]

    def get_n_source(self):
        r"""Gets number of source domain samples."""
        return len(self.source_features)

    def get_n_target(self):
        r"""Gets number of target domain samples."""
        return len(self.target_features)

    def get_source_labels(self, one_hot=False):
        r"""Get labels corresponding to source domain(s)."""
        if one_hot:
            return self.source_labels.numpy()
        else:
            return self.source_labels.argmax(dim=1).numpy()

    def __len__(self):
        r"""Gets number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        src_idx = idx[0]
        tgt_idx = idx[1]

        xs = self.source_features[src_idx]
        ys = self.source_labels[src_idx]
        xt = self.target_features[tgt_idx]
        ret = [xs, ys, xt]

        if self.target_labels is not None:
            ret.append(self.target_labels[tgt_idx])
        if self.target_pseudo_labels is not None:
            ret.append(self.target_pseudo_labels[tgt_idx])
        return ret


class FeaturesMSDADataset(torch.utils.data.Dataset):
    r"""Multi Source Domain Adaptation (MSDA) tabular dataset. This class
    receives (Xs, Ys, Xt), where Xs is a list of tensors containing each
    source domain feature matrix, Ys is a list of tensors containing the
    source domain label matrix, and Xt is a tensor containing the target
    domain feature matrix. Given a list of indices (is1, ..., isN, iT),
    this dataset returns ([Xs1[is1, :], ..., XsN[isN, :]],
    [Ys1[is1, :], ..., YsN[isN, :]], Xt[iT, :]).

    Parameters
    ----------
    source_features : list of tensors
        Tensor of shape (ns, d) containing the feature vectors (in Rd) of
        source domain samples.
    source_labels : list of tensors
        Tensor of shape (ns, nc) containing the one-hot encoded labels of
        source domain feature vectors.
    target_features : tensor
        Tensor of shape (n, d) containing the feature vectors (in Rd) of
        target domain samples.
    target_labels : tensor, optional (default=None)
        Tensor of shape (n, nc) containing the one-hot encoded labels of
        target domain feature vectors.
    target_pseudo_labels : tensor, optional (default=None)
        Tensor of shape (n, nc) containing the pseudo-labels of target
        domain feature vectors. These pseudo-labels correspond to
        probabilities of a sample belonging to a given class.
    """
    def __init__(self,
                 source_features,
                 source_labels,
                 target_features,
                 target_labels=None,
                 target_pseudo_labels=None):
        self.source_features = source_features
        self.source_labels = source_labels
        self.target_features = target_features
        self.target_labels = target_labels
        self.target_pseudo_labels = target_pseudo_labels

        self.num_domains = len(self.source_features)
        self.num_samples = np.min(
            [len(src_fts) for src_fts in self.source_features] +
            [len(self.target_features)]
        )
        self.num_classes = self.source_labels[0].shape[1]
        self.num_features = self.source_features[0].shape[1]

    def get_n_source(self):
        r"""Gets number of source domain samples."""
        return [len(xs) for xs in self.source_features]

    def get_n_target(self):
        r"""Gets number of target domain samples."""
        return len(self.target_features)

    def get_source_labels(self, one_hot=False):
        r"""Get labels corresponding to source domain(s)."""
        if one_hot:
            return [_l.numpy() for _l in self.source_labels]
        else:
            return [_l.argmax(dim=1).numpy() for _l in self.source_labels]

    def __len__(self):
        r"""Gets number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        src_ind = idx[:-1]
        tgt_idx = idx[-1]

        xs = [xs[src_idx]
              for xs, src_idx in zip(self.source_features, src_ind)]
        ys = [ys[src_idx]
              for ys, src_idx in zip(self.source_labels, src_ind)]
        xt = self.target_features[tgt_idx]
        ret = [xs, ys, xt]

        if self.target_labels is not None:
            ret.append(self.target_labels[tgt_idx])
        if self.target_pseudo_labels is not None:
            ret.append(self.target_pseudo_labels[tgt_idx])
        return ret
