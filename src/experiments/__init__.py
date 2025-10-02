import os
from functools import partial
from typing import Optional, Literal, Callable, Any, Union, overload, cast

import torch
from sklearn import metrics
import pandas as pd
from torch.utils.data import Dataset

import medmnist_dataset

import medmnist as medmnist_data

from data import MappedDataset, CachedDataset, IndexedDataset
from torchutil import TorchDevice

from common import Movable

import numpy as np

image_size: int = 64


datasets: list[medmnist_dataset.MedMNIST2D] = []

for mnist in sorted(medmnist_dataset.MedMNIST2D, key=lambda x: x.dataset_sizes['train']):
    if (mnist.is_task_classification and not mnist.is_multi_label or
            mnist.task == medmnist_dataset.MLTask.ordinal_regression):
        datasets.append(mnist)

reported_metrics = [
    ('Accuracy', metrics.accuracy_score),
    ('Balanced accuracy', metrics.balanced_accuracy_score),
    ('Micro F1', partial(metrics.f1_score, average='micro')),
    ('Micro Precision', partial(metrics.precision_score, average='micro')),
    ('Micro Recall', partial(metrics.recall_score, average='micro')),
    ('Macro F1', partial(metrics.f1_score, average='macro')),
    ('Macro Precision', partial(metrics.precision_score, average='macro')),
    ('Macro Recall', partial(metrics.recall_score, average='macro'))
]

reported_metric_scores = [
    ('Micro AUROC', partial(metrics.roc_auc_score, average='micro', multi_class='ovr')),
    ('Macro AUROC', partial(metrics.roc_auc_score, average='macro', multi_class='ovr'))
]


def to_dataframe(results: list[list[float]]) -> pd.DataFrame:
    return (pd.DataFrame(results,
                         index=[ds.display_text for ds in datasets][:len(results)],
                         columns=[metric for metric, _ in reported_metrics + reported_metric_scores])
            .dropna(axis='columns', how='all'))


debug_class_size = 2


@overload
def prepare_datasets(medmnist_dir: str,
                     dataset: medmnist_dataset.MedMNIST2D,
                     require_validation_sets: Literal[False],
                     device: TorchDevice = None,
                     cache_dir: Optional[str] = None,
                     transform: Literal[None] = None,
                     uncached_transform: Literal[None] = None,
                     column_names: tuple[str, ...] = ('images', 'labels'),
                     debug: bool = False) \
        -> tuple[medmnist_data.dataset.MedMNIST2D, medmnist_data.dataset.MedMNIST2D]:
    ...


@overload
def prepare_datasets(medmnist_dir: str,
                     dataset: medmnist_dataset.MedMNIST2D,
                     require_validation_sets: Literal[True],
                     device: TorchDevice = None,
                     cache_dir: Optional[str] = None,
                     transform: Literal[None] = None,
                     uncached_transform: Literal[None] = None,
                     column_names: tuple[str, ...] = ('images', 'labels'),
                     debug: bool = False) \
        -> tuple[medmnist_data.dataset.MedMNIST2D, medmnist_data.dataset.MedMNIST2D, medmnist_data.dataset.MedMNIST2D]:
    ...


@overload
def prepare_datasets(medmnist_dir: str,
                     dataset: medmnist_dataset.MedMNIST2D,
                     require_validation_sets: Literal[False],
                     device: TorchDevice = None,
                     cache_dir: Optional[str] = None,
                     transform: Optional[Callable[[Any], tuple[torch.Tensor, ...]]] = None,
                     uncached_transform: Optional[Callable[[tuple[torch.Tensor, ...]], Movable]] = None,
                     column_names: tuple[str, ...] = ('images', 'labels'),
                     debug: bool = False) \
        -> tuple[Dataset, Dataset]:
    ...


@overload
def prepare_datasets(medmnist_dir: str,
                     dataset: medmnist_dataset.MedMNIST2D,
                     require_validation_sets: Literal[True],
                     device: TorchDevice = None,
                     cache_dir: Optional[str] = None,
                     transform: Optional[Callable[[Any], tuple[torch.Tensor, ...]]] = None,
                     uncached_transform: Optional[Callable[[tuple[torch.Tensor, ...]], Movable]] = None,
                     column_names: tuple[str, ...] = ('images', 'labels'),
                     debug: bool = False) \
        -> tuple[Dataset, Dataset, Dataset]:
    ...


def prepare_datasets(medmnist_dir: str,
                     dataset: medmnist_dataset.MedMNIST2D,
                     require_validation_sets: bool,
                     device: TorchDevice = None,
                     cache_dir: Optional[str] = None,
                     transform: Optional[Callable[[Any], tuple[torch.Tensor, ...]]] = None,
                     uncached_transform: Optional[Callable[[tuple[torch.Tensor, ...]], Movable]] = None,
                     column_names: tuple[str, ...] = ('images', 'labels'),
                     debug: bool = False) \
        -> Union[tuple[Dataset, Dataset], tuple[Dataset, Dataset, Dataset]]:
    if cache_dir is not None and transform is None:
        raise ValueError('`cache_dir` must only be set to a value if `transform` is set to a value.')

    splits = [medmnist_dataset.DatasetSplit.train]

    if require_validation_sets:
        splits.append(medmnist_dataset.DatasetSplit.val)

    splits.append(medmnist_dataset.DatasetSplit.test)

    datasets = medmnist_dataset.initialize_mnist(
        dataset.get_subset(image_size),
        os.path.join(medmnist_dir, dataset.dataset_name),
        *splits,
        convert_to_numpy=True, channel_last=False, download_to_root_directory=True)

    if debug:
        def reduce_dataset(dataset: medmnist_data.dataset.MedMNIST2D) -> Dataset:
            labels = cast(np.ndarray, dataset.labels).flatten()
            classes = np.unique(labels)

            indices = []

            # for each class, take only up to `debug_class_size` items
            for c in classes:
                class_indices = np.argwhere(labels == c)
                indices.append(class_indices[:debug_class_size])

            indices = np.concatenate(indices, axis=0).flatten()

            return IndexedDataset(dataset, indices)

        datasets = tuple(reduce_dataset(ds) for ds in datasets)

    if transform is not None:
        if cache_dir is None or debug:
            datasets = tuple(MappedDataset(ds, transform, device=device,
                                           uncached_transform=uncached_transform)
                             for ds in datasets)
        else:
            cached = []
            for ds, split in zip(datasets, splits):
                files = tuple(os.path.join(cache_dir, split.name, col + '.nd') for col in column_names)

                cached.append(CachedDataset(files, ds, device, transform, use_tqdm=True,
                                            uncached_transform=uncached_transform))

            datasets = tuple(cached)

    return datasets


def training_set_label_proportions(medmnist_dir: str, dataset: medmnist_dataset.MedMNIST2D) -> torch.Tensor:
    ds = medmnist_dataset.initialize_mnist(dataset.get_subset(image_size),
                                           os.path.join(medmnist_dir, dataset.dataset_name),
                                           'train',
                                           download_to_root_directory=True,
                                           convert_to_numpy=False,
                                           unpack_singleton=True)

    assert isinstance(ds, medmnist_data.dataset.MedMNIST2D)

    cls, counts = np.unique(ds.labels, return_counts=True)

    prop = np.arange(cls.max() + 1)
    prop[cls] = counts

    return torch.tensor(prop)
