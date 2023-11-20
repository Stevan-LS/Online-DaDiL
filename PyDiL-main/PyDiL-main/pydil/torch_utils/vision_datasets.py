r"""Vision Datsaset definitions

Implemented datasets: Office31 [1], Office-Home [2], Adaptiope [3],
DomainNet [4].

References

[1] Saenko, K., Kulis, B., Fritz, M., & Darrell, T. (2010). Adapting visual
category models to new domains. In Computer Vision–ECCV 2010: 11th European
Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010,
Proceedings, Part IV 11 (pp. 213-226). Springer Berlin Heidelberg.

[2] Venkateswara, H., Eusebio, J., Chakraborty, S., & Panchanathan, S. (2017).
Deep hashing network for unsupervised domain adaptation. In Proceedings of the
IEEE conference on computer vision and pattern recognition (pp. 5018-5027).

[3] Ringwald, T., & Stiefelhagen, R. (2021). Adaptiope: A modern benchmark for
unsupervised domain adaptation. In Proceedings of the IEEE/CVF Winter
Conference on Applications of Computer Vision (pp. 101-110).

[4] Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019).
Moment matching for multi-source domain adaptation. In Proceedings of the
IEEE/CVF international conference on computer vision (pp. 1406-1415).

"""


import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.nn import functional as F

from pydil.torch_utils.dataset_definitions import all_domains
from pydil.torch_utils.dataset_definitions import all_datasets
from pydil.torch_utils.dataset_definitions import all_definitions


class ObjectRecognitionDataset(torch.utils.data.Dataset):
    r"""Object Recognition dataset. This class implements different object
    recognition domain adaptation benchmarks. Implemented datasets: Office31
    of [1], Office-Home of [2], Adaptiope of [3], DomainNet of [4].

    Parameters
    ----------
    root : str
        Path towards the root of dataset.
    dataset_name : str, optional (default='office31')
        Name of selected dataset.
    domains : list of str
        List of domains in the dataset.
    transform : torchvision transforms
        List of transformations to apply to images
    train : bool, optional (default=True)
        Include training samples in the dataset
    test : bool, optional (default=True)
        Include test samples in the dataset
    multi_source : bool, optional (default=True)
        If True, considers that the dataset consists of multiple source
        domains. This implies that batches are separated according the domain.
    """

    def __init__(self,
                 root,
                 dataset_name='office31',
                 domains=None,
                 transform=None,
                 train=True,
                 test=True,
                 multi_source=True):
        assert dataset_name in all_datasets, (f"Dataset {dataset_name} not"
                                              f" implemented.")

        self.dataset_name = dataset_name
        self.name2cat = all_definitions[dataset_name]
        self.all_domains = all_domains[dataset_name]

        self.root = root
        self.folds_path = os.path.join(root, 'folds')
        if multi_source:
            default = self.all_domains[:-1]
        else:
            default = [self.all_domains[0]]
        self.domains = domains if domains is not None else default
        self.transform = transform
        self.train = train
        self.test = test
        self.multi_source = multi_source
        self.num_classes = len(self.name2cat)

        if multi_source:
            self.filepaths = {}
            self.labels = {}

            for domain in self.domains:
                (filepaths,
                 labels) = self.__get_filenames_and_labels(domain,
                                                           train=train,
                                                           test=test)

                self.filepaths[domain] = filepaths
                self.labels[domain] = labels
        else:
            self.filepaths = []
            self.labels = []

            for domain in self.domains:
                (filepaths,
                 labels) = self.__get_filenames_and_labels(domain,
                                                           train=train,
                                                           test=test)

                self.filepaths.append(filepaths)
                self.labels.append(labels)
            self.filepaths = np.concatenate(self.filepaths)
            self.labels = np.concatenate(self.labels)

    def __str__(self):
        r"""Returns a string representing the dataset."""
        return (f"Dataset {self.dataset_name} with domains {self.domains}"
                f"  and {len(self)} samples")

    def __repr__(self):
        r"""String representation for the dataset"""
        return str(self)

    def __get_filenames_and_labels(self, dom, train=True, test=True):
        r"""Get filenames and labels.

        Parameters
        ----------
        dom : str
            Domain for which filenames and labels will be acquired.
        train : bool, optional (default=True)
            If True, includes train samples in the filenames and labels.
        test : bool, optional (default=True)
            If True, includes test samples in the filenames and labels.
        """
        filepaths, labels = [], []

        class_and_filenames = []

        train_path = os.path.join(self.folds_path,
                                  f'{dom}_train_filenames.txt')
        with open(train_path, 'r') as file:
            train_filenames = file.read().split('\n')[:-1]

        if train:
            class_and_filenames += train_filenames

        test_path = os.path.join(self.folds_path,
                                 f'{dom}_test_filenames.txt')
        with open(test_path, 'r') as file:
            test_filenames = file.read().split('\n')[:-1]

        if test:
            class_and_filenames += test_filenames

        classes = [fname.split('/')[0] for fname in class_and_filenames]
        filenames = [fname.split('/')[1] for fname in class_and_filenames]

        for c, fname in zip(classes, filenames):
            filepaths.append(os.path.join(
                self.root, dom, c, fname))
            labels.append(self.name2cat[c])

        filepaths = np.array(filepaths)
        labels = F.one_hot(torch.from_numpy(
            np.array(labels)).long(), num_classes=self.num_classes).float()

        return filepaths, labels

    def __len__(self):
        r"""Returns the number of samples in the dataset"""
        if self.multi_source:
            return min([
                len(self.filepaths[domain]) for domain in self.filepaths])
        else:
            return len(self.filepaths)

    def __process_index_single_source(self, idx):
        r"""Returns an image and a label corresponding to the index idx."""
        x, y = Image.open(self.filepaths[idx]), self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __process_indices_single_source(self, idx):
        r"""Returns a batch of images and labels corresponding to the
        indices in idx."""
        x, y = [], []
        for i in idx:
            _x, _y = Image.open(self.filepaths[i]), self.labels[i]

            if self.transform:
                _x = self.transform(_x)
            x.append(_x)
            y.append(_y)
        return torch.stack(x), torch.stack(y)

    def __process_index_multi_source(self, idx):
        r"""Returns a list of images corresponding to the image and label
        idx of each domain."""
        x, y = [], []
        for domain in self.domains:
            _x = Image.open(self.filepaths[domain][idx])
            if self.transform:
                _x = self.transform(_x)
            _y = self.labels[domain][idx]
            x.append(_x)
            y.append(_y)
        return x, y

    def __process_indices_multi_source(self, idx):
        r"""Returns a list of batches of images and labels corresponding to
        the indices in idx."""
        x, y = [], []
        for domain, inds in zip(self.domains, idx):
            _x = Image.open(self.filepaths[domain][inds])
            if self.transform:
                _x = self.transform(_x)
            _y = self.labels[domain][inds]

            x.append(_x)
            y.append(_y)
        return x, y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            if self.multi_source:
                return self.__process_indices_multi_source(idx)
            else:
                return self.__process_indices_single_source(idx)
        else:
            if self.multi_source:
                return self.__process_index_multi_source(idx)
            else:
                return self.__process_index_single_source(idx)


class ObjectRecognitionDADataset(torch.utils.data.Dataset):
    r"""Object Recognition dataset for Domain Adaptation. This class implements
    different object recognition domain adaptation benchmarks. Implemented
    datasets: Office31 of [1], Office-Home of [2], Adaptiope of [3],
    DomainNet of [4].

    Parameters
    ----------
    root : str
        Path towards the root of dataset.
    dataset_name : str, optional (default='office31')
        Name of selected dataset.
    source_domain : list of str, optional (default=None)
        List of source domains. If not given, uses all domains except
        last as sources.
    target_domain : str
        Name of target domain. If not given, uses last domain as
        the target.
    transform : torchvision transforms
        List of transformations to apply to images
    train : bool, optional (default=True)
        Include training samples in the dataset
    test : bool, optional (default=True)
        Include test samples in the dataset
    multi_source : bool, optional (default=True)
        If True, considers that the dataset consists of multiple source
        domains. This implies that batches are separated according the domain.
    """

    def __init__(self,
                 root,
                 dataset_name='office31',
                 source_domains=None,
                 target_domain=None,
                 transform=None,
                 train=True,
                 test=True,
                 multi_source=True):
        assert dataset_name in all_datasets, (f"Dataset {dataset_name} not"
                                              f" implemented.")

        self.dataset_name = dataset_name
        self.name2cat = all_definitions[dataset_name]
        self.all_domains = all_domains[dataset_name]

        self.root = root
        self.folds_path = os.path.join(root, 'folds')
        if multi_source:
            default_src = self.all_domains[:-1]
        else:
            default_src = [self.all_domains[0]]

        if source_domains is not None:
            self.source_domains = source_domains
        else:
            self.source_domains = default_src

        if target_domain is not None:
            self.target_domain = target_domain
        else:
            self.target_domain = self.all_domains[-1]

        if self.target_domain in self.source_domains:
            raise ValueError(f"Domain leakage, {self.target_domain} ∈"
                             f" {self.source_domains}")

        self.transform = transform
        self.train = train
        self.test = test
        self.multi_source = multi_source
        self.num_classes = len(self.name2cat)

        if multi_source:
            self.filepaths = {}
            self.labels = {}

            for src in self.source_domains:
                (filepaths,
                 labels) = self.__get_filenames_and_labels(src,
                                                           train=train,
                                                           test=test)

                self.filepaths[src] = filepaths
                self.labels[src] = labels

            (filepaths,
             labels) = self.__get_filenames_and_labels(self.target_domain,
                                                       train=True,
                                                       test=False)
            self.filepaths[self.target_domain] = filepaths
            self.labels[self.target_domain] = labels
        else:
            self.filepaths = {}
            self.labels = {}

            src_f, src_l = [], []
            for src in self.source_domains:
                (filepaths,
                 labels) = self.__get_filenames_and_labels(src,
                                                           train=train,
                                                           test=test)

                src_f.append(filepaths)
                src_l.append(labels)
            self.filepaths['source'] = np.concatenate(src_f)
            self.labels['source'] = np.concatenate(src_l)
            (tgt_f,
             tgt_l) = self.__get_filenames_and_labels(self.target_domain,
                                                      train=True,
                                                      test=False)
            self.filepaths['target'] = tgt_f
            self.labels['target'] = tgt_l

    def get_n_source(self):
        r"""Gets number of source domain samples."""
        if self.multi_source:
            return [len(self.filepaths[src]) for src in self.source_domains]
        else:
            return len(self.filepaths['source'])

    def get_n_target(self):
        r"""Gets number of target domain samples."""
        if self.multi_source:
            return len(self.filepaths[self.target_domain])
        else:
            return len(self.filepaths['target'])

    def get_source_labels(self, one_hot=False):
        r"""Get labels corresponding to source domain(s)."""
        if self.multi_source:
            if one_hot:
                return [
                    self.labels[src].numpy()
                    for src in self.source_domains]
            else:
                return [
                    self.labels[src].argmax(dim=1).numpy()
                    for src in self.source_domains]
        else:
            if one_hot:
                return self.labels['source'].numpy()
            else:
                return self.labels['source'].argmax(dim=1).numpy()

    def __str__(self):
        r"""Returns a string representing the dataset."""
        return (f"Dataset {self.dataset_name} with sources"
                f" {self.source_domains}, target {self.target_domain}"
                f"  and {len(self)} samples")

    def __repr__(self):
        r"""String representation for the dataset"""
        return str(self)

    def __get_filenames_and_labels(self, dom, train=True, test=True):
        r"""Get filenames and labels.

        Parameters
        ----------
        dom : str
            Domain for which filenames and labels will be acquired.
        train : bool, optional (default=True)
            If True, includes train samples in the filenames and labels.
        test : bool, optional (default=True)
            If True, includes test samples in the filenames and labels.
        """
        filepaths, labels = [], []

        class_and_filenames = []

        train_path = os.path.join(self.folds_path,
                                  f'{dom}_train_filenames.txt')
        with open(train_path, 'r') as file:
            train_filenames = file.read().split('\n')[:-1]

        if train:
            class_and_filenames += train_filenames

        test_path = os.path.join(self.folds_path,
                                 f'{dom}_test_filenames.txt')
        with open(test_path, 'r') as file:
            test_filenames = file.read().split('\n')[:-1]

        if test:
            class_and_filenames += test_filenames

        classes = [fname.split('/')[0] for fname in class_and_filenames]
        filenames = [fname.split('/')[1] for fname in class_and_filenames]

        for c, fname in zip(classes, filenames):
            filepaths.append(os.path.join(
                self.root, dom, c, fname))
            labels.append(self.name2cat[c])

        filepaths = np.array(filepaths)
        labels = F.one_hot(torch.from_numpy(
            np.array(labels)).long(), num_classes=self.num_classes).float()

        return filepaths, labels

    def __len__(self):
        r"""Returns the number of samples in the dataset"""
        if self.multi_source:
            return min([
                len(self.filepaths[domain]) for domain in self.filepaths])
        else:
            return min([len(self.filepaths['source']),
                        len(self.filepaths['target'])])

    def __process_index_single_source(self, idx):
        r"""Returns (xs, ys, xt), where xs is the source domain image, ys
        is its corresponding label and xt is the target domain image."""
        inds, indt = idx
        xs = Image.open(self.filepaths['source'][inds])
        xt = Image.open(self.filepaths['target'][indt])
        ys = self.labels['source'][inds]

        if self.transform:
            xs = self.transform(xs)
            xt = self.transform(xt)

        return xs, ys, xt

    def __process_indices_single_source(self, idx):
        r"""Returns (xs, ys, xt), where xs is a bach of source domain images,
        ys is its corresponding labels and xt is a batch of target domain
        images."""
        xs, ys, xt = [], [], []
        for i in idx:
            _xs = Image.open(self.filepaths['source'][i])
            _ys = self.labels['source'][i]
            _xt = Image.open(self.filepaths['target'][i])

            if self.transform:
                _xs = self.transform(_xs)
                _xt = self.transform(_xt)
            xs.append(_xs)
            xt.append(_xt)
            ys.append(_ys)
        return torch.stack(xs), torch.stack(ys), torch.stack(xt)

    def __process_index_multi_source(self, idx):
        r"""Returns (xs, ys, xt), where xs is a list of images from each source
        domain, ys is a list of its corresponding labels, and xt is a target
        domain image."""
        xs, ys = [], []
        inds, idx_tgt = idx[:-1], idx[-1]
        for src, idx_src in zip(self.source_domains, inds):
            _xs = Image.open(self.filepaths[src][idx_src])
            if self.transform:
                _xs = self.transform(_xs)
            _ys = self.labels[src][idx_src]
            xs.append(_xs)
            ys.append(_ys)
        xt = Image.open(self.filepaths[self.target_domain][idx_tgt])
        if self.transform:
            xt = self.transform(xt)
        return xs, ys, xt

    def __process_indices_multi_source(self, idx):
        r"""Returns (xs, ys, xt), where xs is a list of batches from each
        source domain, ys is a list of its corresponding labels, and xt is
        a batch of target domain images."""
        xs, ys = [], []
        for src, inds in zip(self.source_domains, idx[:-1]):
            _xs = Image.open(self.filepaths[src][inds])
            if self.transform:
                _xs = self.transform(_xs)
            _ys = self.labels[src][inds]
            xs.append(_xs)
            ys.append(_ys)
        xt = Image.open(self.filepaths[self.target_domain][idx[-1]])
        if self.transform:
            xt = self.transform(xt)
        return xs, ys, xt

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            if self.multi_source:
                return self.__process_indices_multi_source(idx)
            else:
                return self.__process_indices_single_source(idx)
        else:
            if self.multi_source:
                return self.__process_index_multi_source(idx)
            else:
                return self.__process_index_single_source(idx)


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.im_path = os.path.join(
            self.root, 'img_align_celeba', 'img_align_celeba')
        self.filenames = os.listdir(self.im_path)
        self.attrs = (0.5 * (pd.read_csv(os.path.join(self.root,
                      'list_attr_celeba.csv')).values[:, 1:] + 1)).astype(int)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        is_list = False
        if torch.is_tensor(idx):
            idx = idx.tolist()
            is_list = True
        elif type(idx) == list:
            is_list = True

        if is_list:
            x = []
            y = []

            ims = [Image.open(os.path.join(
                self.im_path, self.filenames[i])) for i in idx]
            if self.transform:
                ims = [self.transform(im) for im in ims]
            x = torch.stack(ims)
            y = torch.from_numpy(self.attrs[idx, 1:].astype(int)).long()
        else:
            im = Image.open(os.path.join(self.im_path, self.filenames[idx]))
            if self.transform:
                im = self.transform(im)
            x = im
            y = torch.from_numpy(self.attrs[idx, 1:].astype(int)).long()

        return x, (y + 1) / 2
