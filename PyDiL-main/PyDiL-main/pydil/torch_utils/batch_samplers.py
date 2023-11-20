"""Batch Sampling Module. In this module we implement multiple
batch sampling strategies, depending on whether we want the same
number of classes per batch (balanced sampling) or not."""

import torch
import numpy as np


class UnbalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Unbalaced Batch Sampler for standard datasets, that is, datasets
    that do not involve domain adaptation.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    batch_size : int
        Number of samples per batch.
    n_bathes : int, optional (default=None)
        Number of batches sampled per epoch. If given, halts epoch at
        n_batches.
    shuffle_indices : bool, optional (default=True)
        If True, shuffles indices after each epoch.
    """
    def __init__(self,
                 n_samples,
                 batch_size,
                 n_batches=None,
                 shuffle_indices=True):
        self.n_samples = n_samples
        self.batch_size = batch_size
        if n_batches is not None:
            self.n_batches = n_batches
        else:
            self.n_batches = n_samples // batch_size + 1
        self.shuffle_indices = shuffle_indices
        self.indices = np.arange(n_samples)

    def __iter__(self):
        r"""Generates a batch of data."""
        if self.shuffle_indices:
            np.random.shuffle(self.indices)

        batches = [
            self.indices[b * self.batch_size:
                         (b + 1) * self.batch_size]
            for b in range(self.n_batches)
        ]

        for batch in batches:
            yield batch

    def __len__(self):
        r"""Returns number of batches."""
        return self.n_batches


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Balanced Batch Sampler for standard datasets, that is, datasets
    that do not involve domain adaptation.

    Parameters
    ----------
    labels : numpy array
        Labels of samples in the dataset
    samples_per_class : int
        Number of samples per class for batch_size. NOTE: Actual batch size
        will correspond to n_classes times samples_per_class.
    n_bathes : int, optional (default=None)
        Number of batches sampled per epoch. If given, halts epoch at
        n_batches.
    shuffle_indices : bool, optional (default=True)
        If True, shuffles indices after each epoch.
    """
    def __init__(self,
                 labels,
                 samples_per_class,
                 n_batches=None,
                 shuffle_indices=True):
        self.labels = labels
        self.n_samples = len(labels)
        self.shuffle_indices = shuffle_indices
        self.unique_classes = np.unique(labels)
        self.indices = np.arange(self.n_samples)
        self.n_classes = len(self.unique_classes)
        self.spc = samples_per_class
        self.batch_size = samples_per_class * self.n_classes

        if n_batches is not None:
            self.n_batches = n_batches
        else:
            self.n_batches = self.n_samples // self.batch_size + 1

    def __iter__(self):
        r"""Generates a batch of data."""
        if self.shuffle_indices:
            np.random.shuffle(self.indices)

        for b in range(self.n_batches):
            batch = []
            for c in self.unique_classes:
                ind_c = np.where(self.labels == c)[0]
                batch.extend(ind_c[b * self.spc:
                                   (b + 1) * self.spc].tolist())
            yield batch

    def __len__(self):
        r"""Returns number of batches."""
        return self.n_batches


class SSDASampler(torch.utils.data.sampler.BatchSampler):
    """
    Sources:
        https://github.com/kilianFatras/JUMBOT/blob/main/Domain_Adaptation/utils.py
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self,
                 n_source,
                 n_target,
                 n_batches=None,
                 source_labels=None,
                 batch_size=None,
                 samples_per_class=None,
                 balanced=False,
                 shuffle_indices=True):

        self.n_source = n_source
        self.n_target = n_target
        self.balanced = balanced

        self.source_indices = np.arange(n_source)
        self.target_indices = np.arange(n_target)

        self.shuffle_indices = shuffle_indices

        if source_labels is not None:
            self.source_labels = source_labels
            self.unique_classes = np.unique(self.source_labels)
            self.n_classes = len(self.unique_classes)

            if self.balanced:
                self.spc = samples_per_class
                self.batch_size = self.n_classes * self.spc
            else:
                assert batch_size is not None, ("Expected batch_size",
                                                " to be not None.")
                self.batch_size = batch_size
        else:
            if batch_size is None:
                assert n_batches is not None, ("If batch_size isn't specified",
                                               "n_batches should be given.")
                batch_size = n_source // n_batches
            else:
                self.batch_size = batch_size
            assert not balanced, "Expected source_labels to be not None."

        if n_batches is None:
            self.n_batches = np.min([self.n_source // self.batch_size,
                                     self.n_target // self.batch_size])
        else:
            self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle_indices:
            np.random.shuffle(self.source_indices)
            np.random.shuffle(self.target_indices)

        for b in range(self.n_batches):
            if self.balanced:
                src_idx = []
                for c in self.unique_classes:
                    ind_c = np.where(self.source_labels == c)[0]
                    src_idx.extend(ind_c[b * self.spc:
                                         (b + 1) * self.spc].tolist())
                tgt_idx = self.target_indices[b * self.batch_size:
                                              (b + 1) * self.batch_size]
            else:
                src_idx = self.source_indices[b * self.batch_size:
                                              (b + 1) * self.batch_size]
                tgt_idx = self.target_indices[b * self.batch_size:
                                              (b + 1) * self.batch_size]
            yield [indices for indices in zip(src_idx, tgt_idx)]

    def __len__(self):
        return self.n_batches


class MSDASampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self,
                 n_sources,
                 n_target,
                 n_batches=None,
                 source_labels=None,
                 batch_size=None,
                 samples_per_class=None,
                 balanced=False,
                 shuffle_indices=True):
        self.n_target = n_target
        self.balanced = balanced
        self.n_sources = n_sources

        self.source_indices = [np.arange(ns) for ns in self.n_sources]
        self.target_indices = np.arange(n_target)

        self.shuffle_indices = shuffle_indices

        if source_labels is not None:
            self.source_labels = source_labels
            _source_labels = np.concatenate(source_labels, axis=0)
            self.unique_classes = np.unique(_source_labels)
            self.n_classes = len(self.unique_classes)

            if self.balanced:
                self.spc = samples_per_class
                self.batch_size = self.n_classes * self.spc
            else:
                assert batch_size is not None, ("Expected batch_size",
                                                " to be not None.")
                self.batch_size = batch_size
        else:
            if batch_size is None:
                assert n_batches is not None, ("If batch_size isn't specified",
                                               "n_batches should be given.")
                self.batch_size = min(
                    [ns // n_batches for ns in n_sources] +
                    [self.n_target // n_batches])
            else:
                self.batch_size = batch_size
            assert not balanced, "Expected source_labels to be not None."

        if n_batches is None:
            self.n_batches = np.min(
                [ns // self.batch_size for ns in self.n_sources] +
                [self.n_target // self.batch_size])
        else:
            self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle_indices:
            for src_indices in self.source_indices:
                np.random.shuffle(src_indices)
            np.random.shuffle(self.target_indices)

        for b in range(self.n_batches):
            if self.balanced:
                source_indices = []
                for ind_sk, ysk in zip(self.source_indices,
                                       self.source_labels):
                    source_indices_k = []
                    for c in self.unique_classes:
                        ind_sk_c = np.where(ysk == c)[0][b * self.spc:
                                                         (b + 1) * self.spc]
                        source_indices_k.extend(ind_sk[ind_sk_c])
                    source_indices.append(source_indices_k)
                target_indices = self.target_indices[b * self.batch_size:
                                                     (b + 1) * self.batch_size]
            else:
                source_indices = []
                for src_idx in self.source_indices:
                    source_indices.append(
                        src_idx[b * self.batch_size:
                                (b + 1) * self.batch_size]
                    )
                target_indices = self.target_indices[
                    b * self.batch_size: (b + 1) * self.batch_size
                ]
            yield [indices for indices in zip(*source_indices,
                                              target_indices)]

    def __len__(self):
        return self.n_batches


class MultiDomainSampler(torch.utils.data.sampler.BatchSampler):
    """
    Sources:
        https://github.com/kilianFatras/JUMBOT/blob/main/Domain_Adaptation/utils.py
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, n_classes, batch_size, n_batches):
        self.labels = labels
        self.spc = batch_size // n_classes
        self.batch_size = n_classes * self.spc
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            indices = []
            for k, ys_k in enumerate(self.labels):
                selected_indices_k = []
                for c in np.unique(ys_k):
                    ind_c = np.where(ys_k == c)[0]
                    _ind_c = np.random.choice(
                        ind_c, size=self.spc)
                    selected_indices_k.append(_ind_c)
                selected_indices_k = np.concatenate(selected_indices_k, axis=0)
                indices.append(selected_indices_k)
            indices = np.stack(indices).T
            yield indices

    def __len__(self):
        return self.n_batches
