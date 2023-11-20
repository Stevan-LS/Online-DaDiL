import torch
from torchvision.models import ResNet50_Weights

from pydil.torch_utils.vision_datasets import ObjectRecognitionDataset
from pydil.torch_utils.vision_datasets import ObjectRecognitionDADataset

from pydil.torch_utils.batch_samplers import SSDASampler
from pydil.torch_utils.batch_samplers import MSDASampler
from pydil.torch_utils.batch_samplers import UnbalancedBatchSampler
from pydil.torch_utils.batch_samplers import BalancedBatchSampler

from pydil.torch_utils.dataset_definitions import all_datasets


def test_unbalanced_batch_sampler():
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=False,
            transform=T
        )
        S = UnbalancedBatchSampler(n_samples=len(dataset),
                                   batch_size=32,
                                   n_batches=None,
                                   shuffle_indices=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
        batch = next(iter(dataloader))

        assert len(batch[0]) == 32


def test_unbalanced_multi_batch_sampler():
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=False,
            transform=T
        )
        S = UnbalancedBatchSampler(n_samples=len(dataset),
                                   batch_size=32,
                                   n_batches=None,
                                   shuffle_indices=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=S)
        batch = next(iter(dataloader))

        assert len(batch[0]) == 32


def test_unbalanced_ssda_batch_sampler():
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDADataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=False,
            transform=T
        )
        n_source = dataset.get_n_source()
        n_target = dataset.get_n_target()
        source_labels = dataset.labels['source'].argmax(axis=1)
        S = SSDASampler(n_source=n_source,
                        n_target=n_target,
                        n_batches=None,
                        source_labels=source_labels,
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
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDADataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=True,
            transform=T
        )
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
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=False,
            transform=T
        )
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
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDADataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=False,
            transform=T
        )
        n_source = dataset.get_n_source()
        n_target = dataset.get_n_target()
        source_labels = dataset.labels['source'].argmax(axis=1)
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
    for dataset_name in all_datasets:
        T = ResNet50_Weights.IMAGENET1K_V2.transforms()
        dataset = ObjectRecognitionDADataset(
            root=f'/home/efernand/data/{dataset_name}',
            dataset_name=dataset_name,
            multi_source=True,
            transform=T
        )
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
