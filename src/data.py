import os
from functools import partial
from typing import BinaryIO, Any, Callable, Optional, cast, Sized, Iterable, TypeVar, Generic, Union, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from common import Movable

from torchutil import TorchDevice, as_device, torch_dtype_from_int, torch_dtype_to_int
import numpy as np


class FileCorruptError(RuntimeError):
    def __init__(self, *args: Any) -> None:
        if len(args) == 0:
            args = ('The file is corrupt.',)

        super().__init__(*args)


def _read_blob(stream: BinaryIO, size: int) -> bytes:
    blob = stream.read(size)

    if len(blob) != size:
        raise FileCorruptError()

    return blob


def _read_uint64(stream: BinaryIO) -> int:
    return int.from_bytes(_read_blob(stream, 8), byteorder='little')


def _write_uint64(stream: BinaryIO, value: int) -> None:
    stream.write(int.to_bytes(value, 8, byteorder='little'))


@dataclass
class _CacheEntry:
    stream: BinaryIO
    offset: int
    dtype: torch.dtype

    N: int
    entry_shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.entry_shape)

    def check(self, tensor: torch.Tensor) -> None:
        if tuple(tensor.shape) != self.entry_shape:
            raise ValueError('All tensors must have the same shape.')

        if tensor.dtype != self.dtype:
            raise ValueError('All tensors must have the same dtype.')

    @property
    def entry_bytesize(self) -> int:
        return self.entry_size * self.dtype.itemsize

    @property
    def entry_size(self) -> int:
        from math import prod
        return prod(self.entry_shape)

    def stream_offset(self, index: int) -> int:
        return self.entry_bytesize * index + self.offset


def default_transform(value: Any, device: TorchDevice = None) -> tuple[torch.Tensor, ...]:
    assert isinstance(value, tuple) and len(value) == 2

    images, labels = value

    images = torch.tensor(images,
                          device=device,
                          dtype=torch.float32 if np.issubdtype(images.dtype, np.floating) else None)
    labels = torch.tensor(labels, device=device, dtype=torch.int64)

    assert labels.ndim == 0 or labels.ndim == 1 and labels.shape[0] == 1

    if labels.ndim == 0:
        labels = labels.unsqueeze(-1)

    if not images.is_floating_point():
        from torchvision.transforms.v2.functional import to_dtype
        images = to_dtype(images, dtype=torch.float32, scale=True)

    assert images.dtype == torch.float32

    return images, labels


class CachedDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self,
                 files: tuple[str, ...],
                 dataset: Dataset,
                 device: TorchDevice,
                 transform: Callable[[Any], tuple[torch.Tensor, ...]] = ...,
                 uncached_transform: Optional[Callable[[tuple[torch.Tensor, ...]], Movable]] = None,
                 use_tqdm: bool = True) \
            -> None:
        from common import tqdm

        device = as_device(device)

        if transform is ...:
            transform = partial(default_transform, device=device)

        assert hasattr(dataset, '__len__')

        self._files = files
        self._dataset = dataset

        cache_entries = []
        streams: list[BinaryIO] = []

        try:
            for file in files:
                streams.append(open(file, 'rb'))

            for file, stream in zip(files, streams):
                dtype = torch_dtype_from_int(_read_uint64(stream))
                N = _read_uint64(stream)
                entry_ndim = _read_uint64(stream)
                entry_shape = tuple(_read_uint64(stream) for _ in range(entry_ndim))

                if N != len(cast(Sized, dataset)):
                    raise FileCorruptError()

                offset = stream.tell()

                from math import prod
                if stream.seek(0, 2) < N * prod(entry_shape) * dtype.itemsize + offset:
                    # insufficient data in cache
                    raise FileCorruptError()

                cache_entries.append(_CacheEntry(stream, offset, dtype, N, entry_shape))
        except (FileNotFoundError, FileCorruptError) as ex:
            if isinstance(ex, FileCorruptError):
                import warnings
                warnings.warn('The cache file is corrupt or does not correspond to the dataset. Recreating cache.')

            for stream in streams:
                stream.close()

            streams = []
            try:
                for file in files:
                    os.makedirs(os.path.dirname(file), exist_ok=True)
                    streams.append(open(file, 'wb+'))

                cache_entries: list[_CacheEntry] = []

                has_headers = False

                src = dataset
                if use_tqdm:
                    src = tqdm(cast(Iterable, src))

                for data_entry in src:
                    transformed = transform(data_entry)

                    if len(transformed) != len(files):
                        raise RuntimeError('The number of entries in the dataset rows differs from the number of cache '
                                           'files.')

                    for i, (stream, tensor) in enumerate(zip(streams, transformed)):
                        if not isinstance(tensor, torch.Tensor):
                            raise RuntimeError('All columns must be transformed to tensors.')

                        if has_headers:
                            cache_entries[i].check(tensor)
                        else:
                            _write_uint64(stream, torch_dtype_to_int(tensor.dtype))
                            _write_uint64(stream, len(cast(Sized, dataset)))
                            _write_uint64(stream, tensor.ndim)
                            for v in tensor.shape:
                                _write_uint64(stream, v)

                            offset = stream.tell()

                            entry = _CacheEntry(stream, offset, tensor.dtype, len(cast(Sized, dataset)),
                                                tuple(tensor.shape))

                            assert entry.entry_size == tensor.nelement()

                            cache_entries.append(entry)

                        # write raw data to stream
                        if tensor.ndim == 0:
                            tensor = tensor.unsqueeze(-1)

                        stream.write(tensor.contiguous().detach().view(torch.uint8).numpy(force=True).tobytes())

                    has_headers = True

                for stream in streams:
                    stream.flush()
            except:
                for file, stream in zip(files, streams):
                    stream.close()

                    try:
                        os.remove(file)
                    except OSError:
                        pass

                raise

        self._cache_entries = cache_entries
        self._uncached_transform = uncached_transform
        self._device = device

    def _read(self, entry: _CacheEntry, index: int) -> torch.Tensor:
        entry.stream.seek(entry.stream_offset(index))
        blob = entry.stream.read(entry.entry_bytesize)

        return torch.frombuffer(blob,
                                count=entry.entry_size,
                                dtype=entry.dtype,
                                requires_grad=False).to(device=self._device).reshape(entry.entry_shape)

    def __getitem__(self, index: int) -> Movable:
        from common import to
        tensors = tuple(to(self._read(entry, index), device=self._device)
                        for entry in self._cache_entries)

        if self._uncached_transform is not None:
            tensors = self._uncached_transform(tensors)

        return to(tensors, device=self._device)

    def __len__(self) -> int:
        return len(cast(Sized, self._dataset))

    def __del__(self) -> None:
        for cache_entry in self._cache_entries:
            cache_entry.stream.close()


class MappedDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self,
                 dataset: Dataset,
                 transform: Callable[[Any], tuple[torch.Tensor, ...]] = ...,
                 uncached_transform: Optional[Callable[[tuple[torch.Tensor, ...]], Movable]] = None,
                 device: TorchDevice = None) -> None:
        assert hasattr(dataset, '__len__')

        device = as_device(device)

        if transform is ...:
            transform = partial(default_transform, device=device)

        self._uncached_transform = uncached_transform
        self._dataset = dataset
        self._transform = transform
        self._device = device

    def __getitem__(self, item: int) -> Movable:
        from common import to
        values = tuple(t.to(device=self._device)
                       for t in self._transform(self._dataset[item]))

        if self._uncached_transform is not None:
            values = self._uncached_transform(values)

        return to(values, device=self._device)

    def __len__(self) -> int:
        return len(cast(Sized, self._dataset))


_T_co = TypeVar('_T_co', covariant=True)


class IndexedDataset(Dataset[_T_co], Generic[_T_co]):
    def __init__(self, inner: Dataset[_T_co], indices: Union[range, slice, np.ndarray]) -> None:
        if not hasattr(inner, '__len__'):
            raise TypeError('`inner` must have attribute `__len__`.')

        if isinstance(indices, np.ndarray):
            if indices.ndim != 1:
                raise ValueError('`indices` must be a vector.')

        self._inner = inner
        self._indices = indices
        self._getitems: Optional[Callable[[list[int]], list[_T_co]]] = getattr(inner, '__getitems__', None)

    def __getitem__(self, item: int) -> _T_co:
        indices = self._indices

        if isinstance(indices, slice):
            indices = range(*indices.indices(len(cast(Sized, self._inner))))

        return self._inner[indices[item].item()]

    def __getitems__(self, indices: Sequence[int]) -> list[_T_co]:
        if self._getitems is None:
            return [self[i] for i in indices]
        else:
            base_indices = self._indices

            if isinstance(base_indices, slice):
                base_indices = range(*base_indices.indices(len(cast(Sized, self._inner))))

            return [self._getitems([base_indices[i] for i in indices])]

    def __len__(self) -> int:
        return len(self._indices)
