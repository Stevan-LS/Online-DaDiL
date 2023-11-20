import torch
import numpy as np
from tqdm.auto import tqdm
from pydil.ipms.mmd_ipms import ClassConditionalMMD


class DistributionMatching(torch.nn.Module):
    r"""Distribution Matching algorithm of [Zhao and Bilen, 2023].
    This algorithm minimizes the MMD between the original dataset
    and the synthetic summary of samples.

    Parameters
    ----------
    spc : int, optional (default=1)
        Number of samples per class in the data summary.
    n_classes : int, optional (default=2)
        Number of classes in the original dataset.
    n_dim : int, optional (default=2048)
        Number of dimensions in the features of original dataset.
    xsyn : Tensor, optional (default=None)
        If given, initializes synthetic summary features with
        the given tensor.
    ysyn : Tensor, optional (default=None)
        If given, initializes synthetic summary labels with
        the given tensor.
    loss_fn : function, optional (default=None)
        Function minimizes throughout optimization. If not given,
        uses the ClassConditionalMMD as in the original paper.
    optimizer_name : str, optional (default='sgd')
        Name of optimizer to be used. Either 'adam' or 'sgd'.
    learning_rate : float, optional (default=1e-1)
        Value for learning rate.
    momentum : float, optional (default=0.9)
        Value for momentum. Only used in SGD.
    verbose : bool, optional (default=False)
        If True, prints loss value at each iteration.
    """
    def __init__(self,
                 spc=1,
                 n_classes=2,
                 n_dim=2048,
                 xsyn=None,
                 ysyn=None,
                 loss_fn=None,
                 optimizer_name='sgd',
                 learning_rate=1e-1,
                 momentum=0.9,
                 verbose=False):
        super(DistributionMatching, self).__init__()
        self.verbose = verbose

        if ysyn is None:
            self.ipc = spc
            self.n_classes = n_classes
            ysyn = torch.cat([torch.tensor([k] * spc)
                              for k in range(self.n_classes)]).long()
            ysyn = torch.nn.functional.one_hot(ysyn,
                                               num_classes=n_classes).float()
        else:
            self.n_classes = ysyn.shape[1]
            self.ipc = ysyn.shape[0] // ysyn.shape[1]

        if xsyn is None:
            self.n_dim = n_dim
            xsyn = torch.randn(self.ipc * self.n_classes, self.n_dim)
        else:
            self.n_dim = xsyn.shape[1]

        if loss_fn is None:
            self.loss_fn = ClassConditionalMMD()
        else:
            self.loss_fn = loss_fn

        self.ysyn = ysyn
        self.xsyn = torch.nn.Parameter(data=xsyn, requires_grad=True)

        self.optimizer = self.configure_optimizer(optimizer_name,
                                                  learning_rate,
                                                  momentum)

    def configure_optimizer(self,
                            optimizer_name,
                            learning_rate,
                            momentum):
        """Returns the optimizer for distillation"""
        if optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(),
                                   lr=learning_rate,
                                   momentum=momentum)
        else:
            return torch.optim.Adam(self.parameters(),
                                    lr=learning_rate)

    def fit(self, X, Y, batch_size=256, n_iter=100, quiet=True):
        """Dataset Distillation function.

        Parameters
        ----------
        X : Tensor
            Tensor of shape (n, d) consisting of original dataset
            features.
        Y : Tensor
            Tensor of shape (n, nc) consisting of original dataset
            labels.
        batch_size : int, optional (default=256)
            Number of elements in batches from the original dataset
        n_iter : int, optional (default=100)
            Number of iterations in distillation
        quiet : bool, optional (default=True)
            If False, plots progress bar at each iteration.
        """
        tr_dataset = torch.utils.data.TensorDataset(X, Y)
        tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=False)

        history = []
        for it in range(n_iter):
            if quiet:
                pbar = tr_loader
            else:
                pbar = tqdm(tr_loader)
            it_loss = 0.0
            for x, y in pbar:
                self.optimizer.zero_grad()
                loss = self.loss_fn(x, y, self.xsyn, self.ysyn)

                loss.backward()
                self.optimizer.step()

                it_loss += loss.item() / len(tr_loader)
            if self.verbose:
                print(f'It {it}, Loss {it_loss}')
            history.append(it_loss)

        return history


class RegularizedDistributionMatching(torch.nn.Module):
    r"""MSDA adaptation of the Distribution Matching algorithm of
    [Zhao and Bilen, 2023]. This algorithm minimizes the MMD between
    the original dataset and the synthetic summary of samples.

    Parameters
    ----------
    spc : int, optional (default=1)
        Number of samples per class in the data summary.
    n_classes : int, optional (default=2)
        Number of classes in the original dataset.
    n_dim : int, optional (default=2048)
        Number of dimensions in the features of original dataset.
    xsyn : Tensor, optional (default=None)
        If given, initializes synthetic summary features with
        the given tensor.
    ysyn : Tensor, optional (default=None)
        If given, initializes synthetic summary labels with
        the given tensor.
    loss_fn : function, optional (default=None)
        Function minimizes throughout optimization. If not given,
        uses the ClassConditionalMMD as in the original paper.
    optimizer_name : str, optional (default='sgd')
        Name of optimizer to be used. Either 'adam' or 'sgd'.
    learning_rate : float, optional (default=1e-1)
        Value for learning rate.
    momentum : float, optional (default=0.9)
        Value for momentum. Only used in SGD.
    verbose : bool, optional (default=False)
        If True, prints loss value at each iteration.
    """
    def __init__(self,
                 spc=1,
                 n_classes=2,
                 n_dim=2048,
                 xsyn=None,
                 ysyn=None,
                 loss_fn=None,
                 optimizer_name='sgd',
                 learning_rate=1e-1,
                 momentum=0.9,
                 verbose=False):
        super(RegularizedDistributionMatching, self).__init__()
        self.verbose = verbose

        if ysyn is None:
            self.ipc = spc
            self.n_classes = n_classes
            ysyn = torch.cat([torch.tensor([k] * spc)
                              for k in range(self.n_classes)]).long()
            ysyn = torch.nn.functional.one_hot(ysyn,
                                               num_classes=n_classes).float()
        else:
            self.n_classes = ysyn.shape[1]
            self.ipc = ysyn.shape[0] // ysyn.shape[1]

        if xsyn is None:
            self.n_dim = n_dim
            xsyn = torch.randn(self.ipc * self.n_classes, self.n_dim)
        else:
            self.n_dim = xsyn.shape[1]

        if loss_fn is None:
            self.loss_fn = ClassConditionalMMD()
        else:
            self.loss_fn = loss_fn

        self.ysyn = ysyn
        self.xsyn = torch.nn.Parameter(data=xsyn, requires_grad=True)

        self.optimizer = self.configure_optimizer(optimizer_name,
                                                  learning_rate,
                                                  momentum)

    def configure_optimizer(self,
                            optimizer_name,
                            learning_rate,
                            momentum):
        """Returns the optimizer for dataset distillation."""
        if optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(),
                                   lr=learning_rate,
                                   momentum=momentum)
        else:
            return torch.optim.Adam(self.parameters(),
                                    lr=learning_rate)

    def fit(self, Xs, Ys, Xt, batch_size=256, n_iter=100, batches_per_it=10):
        """Dataset Distillation function.

        Parameters
        ----------
        Xs : list of tensors
            List of tensors. Each element is a tensor of shape
            (nk, d), containng the features of each source dataset
        Ys : list of tensors
            List of tensors. Each element is a tensor of shape
            (nk, nc), containng the labels of each source dataset
        Xs : Tensor
            Tensor of shape (nt, d) consisting of target dataset
            features.
        batch_size : int, optional (default=256)
            Number of elements in batches from the original dataset
        n_iter : int, optional (default=100)
            Number of iterations in distillation
        quiet : bool, optional (default=True)
            If False, plots progress bar at each iteration.
        """
        history = []
        src_ind = [np.arange(len(Xsk)) for Xsk in Xs]
        tgt_ind = np.arange(len(Xt))
        for it in range(n_iter):
            it_loss = 0.0
            for _ in range(batches_per_it):
                self.optimizer.zero_grad()

                src_batch_ind = [
                    np.random.choice(ind_k, size=batch_size)
                    for ind_k in src_ind]
                tgt_batch_ind = np.random.choice(tgt_ind, size=batch_size)
                xs = [Xsk[ind_k] for Xsk, ind_k in zip(Xs, src_batch_ind)]
                ys = [Ysk[ind_k] for Ysk, ind_k in zip(Ys, src_batch_ind)]
                xt = Xt[tgt_batch_ind]

                loss = 0
                for xsk, ysk in zip(xs, ys):
                    loss += self.loss_fn(xsk, ysk, self.xsyn, self.ysyn)
                loss += self.loss_fn(xt, None, self.xsyn, None)

                loss.backward()
                self.optimizer.step()

                it_loss += loss.item() / batches_per_it
            if self.verbose:
                print(f'It {it}, Loss {it_loss}')
            history.append(it_loss)

        return history
