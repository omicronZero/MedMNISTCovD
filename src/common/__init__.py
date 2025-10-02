import PIL.Image
import torch
import numpy as np
from typing import Iterable, TypeVar, Sequence, TypeAlias, Union, overload, Iterator, Callable

import torchutil

_T = TypeVar('_T')


def tqdm(iterable: Iterable[_T]) -> Iterable[_T]:
    from tqdm import tqdm
    import sys

    return tqdm(iterable, file=sys.stdout)


def cov_descr(feature_image: torch.Tensor) -> torch.Tensor:
    from torchutil.bcov import bcov
    return bcov(feature_image.flatten(start_dim=-2))


def to_feature_covs(images: np.ndarray) -> torch.Tensor:
    if images.ndim == 3:
        images = np.expand_dims(images, 1)
    assert images.ndim == 4

    N, C, H, W = images.shape

    if C not in (1, 3):
        images = np.moveaxis(images, -1, 1)
        N, C, H, W = images.shape
        assert C in (1, 3)

    covs = []

    for img in tqdm(images):
        fimg = handcrafted_features(img)

        covs.append(cov_descr(torch.tensor(fimg)))

    return torch.stack(covs)


def handcrafted_features(images: np.ndarray) -> np.ndarray:
    import skimage.filters as _feat

    *N, C, H, W = images.shape

    if len(N) > 0:
        images = np.stack([handcrafted_features(img) for img in images.reshape(-1, C, H, W)])
        return images.reshape((*N, images.shape[1:]))

    if C != 1:
        images = np.mean(images, axis=0, keepdims=True)

    images = np.squeeze(images, 0)

    x, y = np.meshgrid(np.arange(W), np.arange(H))

    return np.stack([np.broadcast_to(x, images.shape),
                     np.broadcast_to(y, images.shape),
                     np.abs(_feat.sobel_h(images)),
                     np.abs(_feat.sobel_v(images)),
                     np.abs(_feat.sobel_h(_feat.sobel_h(images))),
                     np.abs(_feat.sobel_v(_feat.sobel_v(images))),
                     np.sqrt(_feat.sobel_h(images) ** 2 + _feat.sobel_v(images) ** 2),
                     np.arctan2(_feat.sobel_v(images), _feat.sobel_h(images))])


def take_samples(values: Sequence[_T], sample_count: int) -> Iterator[_T]:
    perm = np.random.permutation(len(values))[:sample_count].tolist()

    return (values[i] for i in perm)


Movable: TypeAlias = Union[torch.Tensor, tuple['Movable', ...], list['Movable']]


@overload
def to(instance: torch.Tensor,
       device: torchutil.TorchDevice = None,
       dtype: torchutil.TorchDTypeOpt = None) -> torch.Tensor:
    ...


@overload
def to(instance: tuple[torch.Tensor, ...],
       device: torchutil.TorchDevice = None,
       dtype: torchutil.TorchDTypeOpt = None) \
        -> tuple[torch.Tensor, ...]:
    ...


@overload
def to(instance: list[torch.Tensor],
       device: torchutil.TorchDevice = None,
       dtype: torchutil.TorchDTypeOpt = None) -> list[torch.Tensor]:
    ...


@overload
def to(instance: Movable,
       device: torchutil.TorchDevice = None,
       dtype: torchutil.TorchDTypeOpt = None) -> Movable:
    ...


def to(instance: Movable,
       device: torchutil.TorchDevice = None,
       dtype: torchutil.TorchDTypeOpt = None) -> Movable:
    """
    Recursively moves the indicated instances to the specified device and dtype.

    :param instance: The instances to move.
    :param device: The device to move to or `None` to keep the current device.
    :param dtype: The dtype to move to or `None` to keep the current dtype.
    :return: The mapped instances.
    """
    if isinstance(instance, torch.Tensor):
        return instance.to(device=device, dtype=dtype)
    elif isinstance(instance, tuple):
        return tuple(to(v, device, dtype) for v in instance)
    elif isinstance(instance, list):
        return [to(v, device, dtype) for v in instance]

    raise TypeError('`instance` has an unsupported type.')


def pil_to_numpy(img: PIL.Image.Image) -> np.ndarray:
    npa = np.array(img)

    if np.issubdtype(npa.dtype, np.integer):
        npa = npa / np.iinfo(npa.dtype).max

    npa = npa.astype(np.float32)

    if npa.ndim == 2:
        npa = np.expand_dims(npa, 0)

    return npa


def _geomstats_call(f: Callable[[], _T], device: torch.device) -> _T:
    default_dtype = torch.get_default_dtype()

    try:
        torch.set_default_dtype(torch.float64)

        with device:
            return f()
    finally:
        torch.set_default_dtype(default_dtype)


def descriptor_mean(descriptors: torch.Tensor, max_iterations: int = 1000) -> torch.Tensor:
    squeeze = descriptors.ndim == 3

    if squeeze:
        descriptors = descriptors.unsqueeze(0)

    assert descriptors.ndim == 4

    assert descriptors.shape[-1] == descriptors.shape[-2]

    if descriptors.ndim > 3:
        descriptors = descriptors.moveaxis(1, 0)

    Q = descriptors.shape[-1]

    def handle() -> torch.Tensor:
        from typing import cast
        from geomstats.learning.frechet_mean import FrechetMean, BatchGradientDescent
        from geomstats.geometry.spd_matrices import SPDLogEuclideanMetric, SPDMatrices

        manifold = SPDMatrices(Q, equip=False)
        manifold.equip_with_metric(SPDLogEuclideanMetric)

        mean = FrechetMean(manifold)
        mean.__init__(manifold, 'batch')

        cast(BatchGradientDescent, mean.optimizer).max_iter = max_iterations

        mean.fit(descriptors)

        return mean.estimate_.to(dtype=descriptors.dtype, device=descriptors.device)

    result = _geomstats_call(handle, descriptors.device)

    if squeeze:
        result = result.squeeze(0)

    return result


def logeigh(spd: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(spd)

    eigvals = eigvals.log()

    return eigvecs * eigvals.unsqueeze(-2) @ eigvecs.transpose(-1, -2)


def log_euclidean_metric(covs_1: torch.Tensor, covs_2: torch.Tensor) -> torch.Tensor:
    *N1, Q11, Q12 = covs_1.shape
    *N2, Q21, Q22 = covs_2.shape

    assert Q11 == Q12 == Q21 == Q22

    return torch.linalg.matrix_norm(logeigh(covs_1) - logeigh(covs_2))


def log_euclidean_exp(covs: torch.Tensor, basepoint: torch.Tensor) -> torch.Tensor:
    assert covs.device == basepoint.device

    Q = covs.shape[-1]

    rows, columns = torch.triu_indices(covs.shape[-2], covs.shape[-2], device=covs.device)

    def handle() -> torch.Tensor:
        from geomstats.geometry.spd_matrices import SPDMatrices, SPDLogEuclideanMetric
        manif = SPDMatrices(Q, equip=False)

        exp_map = SPDLogEuclideanMetric(manif).exp

        return exp_map(covs, base_point=basepoint)

    mapped = _geomstats_call(handle, covs.device)

    return mapped[..., columns, rows]
