import torch

from pydil.torch_utils.tabular_datasets import FeaturesDataset
from pydil.torch_utils.tabular_datasets import FeaturesSSDADataset
from pydil.torch_utils.tabular_datasets import FeaturesMSDADataset

from pydil.torch_utils.batch_samplers import SSDASampler
from pydil.torch_utils.batch_samplers import MSDASampler
from pydil.torch_utils.batch_samplers import UnbalancedBatchSampler
from pydil.torch_utils.batch_samplers import BalancedBatchSampler


def test_unbalanced_batch_sampler():
    X = torch.randn(1024, 128)
    Y = torch.randn(1024, 10)
    dataset = FeaturesDataset(features=X, labels=Y)
    S = UnbalancedBatchSampler(n_samples=len(dataset),
                               batch_size=32,
                               n_batches=None,
                               shuffle_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
    batch = next(iter(dataloader))

    assert len(batch[0]) == 32
    assert len(batch[1]) == 32


def test_unbalanced_ssda_batch_sampler():
    Xs = torch.randn(1024, 128)
    Ys = torch.randn(1024, 10).softmax(dim=-1)
    Xt = torch.randn(128, 128)
    dataset = FeaturesSSDADataset(source_features=Xs,
                                  source_labels=Ys,
                                  target_features=Xt)
    n_source = dataset.get_n_source()
    n_target = dataset.get_n_target()
    S = SSDASampler(n_source=n_source,
                    n_target=n_target,
                    n_batches=None,
                    source_labels=None,
                    batch_size=32,
                    samples_per_class=None,
                    balanced=False,
                    shuffle_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
    xs, ys, xt = next(iter(dataloader))

    assert xs.shape[0] == 32
    assert ys.shape[0] == 32
    assert xt.shape[0] == 32


def test_unbalanced_msda_batch_sampler():
    Xs = [torch.randn(1024, 128) for _ in range(5)]
    Ys = [torch.randn(1024, 10).softmax(dim=-1) for _ in range(5)]
    Xt = torch.randn(1024, 128)
    dataset = FeaturesMSDADataset(source_features=Xs,
                                  source_labels=Ys,
                                  target_features=Xt)
    n_sources = dataset.get_n_source()
    n_target = dataset.get_n_target()
    S = MSDASampler(n_sources,
                    n_target,
                    n_batches=None,
                    source_labels=None,
                    batch_size=32,
                    samples_per_class=None,
                    balanced=False,
                    shuffle_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
    xs, ys, xt = next(iter(dataloader))

    for xsk, ysk in zip(xs, ys):
        assert xsk.shape[0] == 32
        assert ysk.shape[0] == 32
    assert xt.shape[0] == 32


def test_balanced_batch_sampler():
    X = torch.randn(1024, 128)
    Y = torch.randn(1024, 10).softmax(dim=-1)
    dataset = FeaturesDataset(features=X, labels=Y)
    S = BalancedBatchSampler(labels=dataset.labels.argmax(axis=1),
                             samples_per_class=2,
                             n_batches=None,
                             shuffle_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
    batch = next(iter(dataloader))
    uniques, counts = batch[1].argmax(dim=1).unique(return_counts=True)

    assert torch.eq(uniques, torch.arange(dataset.num_classes)).all()
    assert torch.eq(counts, 2 * torch.ones(dataset.num_classes)).all()
    assert len(batch[0]) == 2 * dataset.num_classes


def test_balanced_ssda_batch_sampler():
    Xs = torch.randn(1024, 128)
    Ys = torch.randn(1024, 10).softmax(dim=-1)
    Xt = torch.randn(128, 128)
    dataset = FeaturesSSDADataset(source_features=Xs,
                                  source_labels=Ys,
                                  target_features=Xt)
    n_source = dataset.get_n_source()
    n_target = dataset.get_n_target()
    source_labels = dataset.get_source_labels()
    S = SSDASampler(n_source=n_source,
                    n_target=n_target,
                    n_batches=None,
                    source_labels=source_labels,
                    batch_size=None,
                    samples_per_class=2,
                    balanced=True,
                    shuffle_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
    batch = next(iter(dataloader))

    xs, ys, xt = batch
    uniques, counts = ys.argmax(dim=1).unique(return_counts=True)

    assert xs.shape[0] == ys.shape[0] == xt.shape[0]
    assert torch.eq(uniques, torch.arange(dataset.num_classes)).all()
    assert torch.eq(counts, 2 * torch.ones(dataset.num_classes)).all()
    assert len(xs) == 2 * dataset.num_classes


def test_balanced_msda_batch_sampler():
    Xs = [torch.randn(1024, 128) for _ in range(5)]
    Ys = [torch.randn(1024, 10).softmax(dim=-1) for _ in range(5)]
    Xt = torch.randn(1024, 128)
    dataset = FeaturesMSDADataset(source_features=Xs,
                                  source_labels=Ys,
                                  target_features=Xt)
    n_sources = dataset.get_n_source()
    n_target = dataset.get_n_target()
    source_labels = dataset.get_source_labels()
    S = MSDASampler(n_sources,
                    n_target,
                    n_batches=None,
                    source_labels=source_labels,
                    batch_size=None,
                    samples_per_class=2,
                    balanced=True,
                    shuffle_indices=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
    xs, ys, xt = next(iter(dataloader))

    for xsk, ysk in zip(xs, ys):
        uk, ck = ysk.argmax(dim=1).unique(return_counts=True)
        assert xsk.shape[0] == 2 * dataset.num_classes
        assert ysk.shape[0] == 2 * dataset.num_classes
        assert torch.eq(uk, torch.arange(dataset.num_classes)).all()
        assert torch.eq(ck, 2 * torch.ones(dataset.num_classes)).all()
    assert xt.shape[0] == 2 * dataset.num_classes
