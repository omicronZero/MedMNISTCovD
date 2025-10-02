from types import MappingProxyType
from typing import Union, Optional, overload, Literal, TypeVar

import numpy as np
import torch

from util import typename

TorchDevice: type = Union[None, str, int, torch.device]
TorchDeviceNoDefault: type = Union[str, int, torch.device]
TorchDType: type = Union[str, torch.dtype, type]
TorchDTypeOpt: type = Union[None, TorchDType]

_np_type_map = {np.bool_: torch.bool,
                np.short: torch.short,
                np.half: torch.half,
                np.double: torch.double,
                np.cdouble: torch.cdouble,
                np.int64: torch.int64,
                np.float16: torch.float16,
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
                np.int32: torch.int32,
                np.int16: torch.int16,
                np.int8: torch.int8,
                np.uint8: torch.uint8,
                torch.bool: np.bool_,
                torch.short: np.short,
                torch.half: np.half,
                torch.double: np.double,
                torch.cdouble: np.cdouble,
                torch.int64: np.int64,
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.complex64: np.complex64,
                torch.complex128: np.complex128,
                torch.int32: np.int32,
                torch.int16: np.int16,
                torch.int8: np.int8,
                torch.uint8: np.uint8}

_type_map = {'uint8': torch.uint8,
             'int8': torch.int8,
             'int16': torch.int16,
             'int32': torch.int32,
             'int64': torch.int64,
             'float16': torch.float16,
             'float32': torch.float32,
             'float64': torch.float64,
             'complex32': torch.complex32,
             'complex64': torch.complex64,
             'complex128': torch.complex128,
             'bool': torch.bool,
             'qint8': torch.qint8,
             'quint8': torch.quint8,
             'qint32': torch.qint32,
             'bfloat16': torch.bfloat16,
             'quint4x2': torch.quint4x2,
             'quint2x4': torch.quint2x4,
             'bits1x8': torch.bits1x8,
             'bits2x4': torch.bits2x4,
             'bits4x2': torch.bits4x2,
             'bits8': torch.bits8,
             'bits16': torch.bits16,
             'float8_e5m2': torch.float8_e5m2,
             'float8_e4m3fn': torch.float8_e4m3fn,
             'float8_e5m2fnuz': torch.float8_e5m2fnuz,
             'float8_e4m3fnuz': torch.float8_e4m3fnuz,
             torch.uint8: 'uint8',
             torch.int8: 'int8',
             torch.int16: 'int16',
             torch.int32: 'int32',
             torch.int64: 'int64',
             torch.float16: 'float16',
             torch.float32: 'float32',
             torch.float64: 'float64',
             torch.complex32: 'complex32',
             torch.complex64: 'complex64',
             torch.complex128: 'complex128',
             torch.bool: 'bool',
             torch.qint8: 'qint8',
             torch.quint8: 'quint8',
             torch.qint32: 'qint32',
             torch.bfloat16: 'bfloat16',
             torch.quint4x2: 'quint4x2',
             torch.quint2x4: 'quint2x4',
             torch.bits1x8: 'bits1x8',
             torch.bits2x4: 'bits2x4',
             torch.bits4x2: 'bits4x2',
             torch.bits8: 'bits8',
             torch.bits16: 'bits16',
             torch.float8_e5m2: 'float8_e5m2',
             torch.float8_e4m3fn: 'float8_e4m3fn',
             torch.float8_e5m2fnuz: 'float8_e5m2fnuz',
             torch.float8_e4m3fnuz: 'float8_e4m3fnuz'}

_torch_dtypes = (torch.quint4x2,
                 torch.quint2x4,
                 torch.bits1x8,
                 torch.bits2x4,
                 torch.bits4x2,
                 torch.bits8,
                 torch.uint8,
                 torch.int8,
                 torch.bool,
                 torch.qint8,
                 torch.quint8,
                 torch.bits16,
                 torch.int16,
                 torch.qint32,
                 torch.int32,
                 torch.int64,
                 torch.float8_e5m2,
                 torch.float8_e4m3fn,
                 torch.float8_e5m2fnuz,
                 torch.float8_e4m3fnuz,
                 torch.bfloat16,
                 torch.float16,
                 torch.float32,
                 torch.float64,
                 torch.complex32,
                 torch.complex64,
                 torch.complex128)

_torch_dtype_to_int = {k: i for i, k in enumerate(_torch_dtypes)}


def torch_dtype_to_int(dtype: TorchDType) -> int:
    return _torch_dtype_to_int[as_dtype(dtype)]


def torch_dtype_from_int(v: int) -> torch.dtype:
    if v < 0:
        raise ValueError('`v` must be non-negative.')

    return _torch_dtypes[v]


def supported_torch_dtypes() -> tuple[torch.dtype, ...]:
    return _torch_dtypes


@overload
def map_torch_dtype(s: str) -> torch.dtype: ...


@overload
def map_torch_dtype(dtype: torch.dtype) -> str: ...


def map_torch_dtype(s_or_dtype: Union[str, torch.dtype, np.dtype]) -> Union[str, torch.dtype]:
    result = _type_map.get(s_or_dtype, None)

    if result is None and isinstance(s_or_dtype, np.dtype):
        result = _np_type_map.get(s_or_dtype, None)

    if result is None:
        raise ValueError(f'The indicated data type is not a supported: \'{s_or_dtype}\' (type: '
                         f'{typename(type(s_or_dtype))}).')

    return result


@overload
def map_torch_np_dtype(dtype: np.dtype) -> torch.dtype: ...


@overload
def map_torch_np_dtype(dtype: torch.dtype) -> np.dtype: ...


def map_torch_np_dtype(dtype: Union[np.dtype, torch.dtype]) -> Union[torch.dtype, np.dtype]:
    return _np_type_map[dtype]


_torch_supported_numpy_dtypes = {}


def _eval_torch_supported_numpy_dtypes() -> None:
    for name, dtype in np.sctypeDict.items():
        if isinstance(name, str):
            torch_dtype = getattr(torch, name, None)
            if torch_dtype is not None and isinstance(torch_dtype, torch.dtype):
                np_dtype = np.dtype(dtype)

                if torch_dtype.itemsize == np_dtype.itemsize:
                    _torch_supported_numpy_dtypes[dtype] = torch_dtype


_eval_torch_supported_numpy_dtypes()

torch_supported_numpy_dtypes = MappingProxyType(_torch_supported_numpy_dtypes)


def as_device(device: TorchDevice,
              none_to_default_device: bool = False,
              resolve_to_default_index: bool = True) -> Optional[torch.device]:
    if device is None:
        if not none_to_default_device:
            return None

        return torch.empty(0).device

    if not isinstance(device, torch.device):
        device = torch.device(device)

    if resolve_to_default_index and device.index is None:
        device = torch.tensor([], device=device).device

    return device


@overload
def as_dtype(dtype: TorchDTypeOpt, resolve_none_to_default: Literal[False]) -> Optional[torch.dtype]: ...


@overload
def as_dtype(dtype: TorchDTypeOpt, resolve_none_to_default: Literal[True]) -> torch.dtype: ...


@overload
def as_dtype(dtype: TorchDTypeOpt, resolve_none_to_default: bool = False) -> Optional[torch.dtype]: ...


def as_dtype(dtype: TorchDTypeOpt, resolve_none_to_default: bool = False) -> Optional[torch.dtype]:
    if dtype is None:
        if resolve_none_to_default:
            return torch.get_default_dtype()

        return None

    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, type):
        dtype = dtype.__name__
    else:
        dtype = str(dtype)

    if isinstance(dtype, str) and dtype.startswith('torch.'):
        dtype = dtype[6:]

    orig_dtype = dtype
    dtype = getattr(torch, dtype, None)

    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f'Unknown torch type for {repr(orig_dtype)}.')

    return dtype


def broadcast_axes(input: torch.Tensor,
                   axes: Union[int, tuple[int, ...]],
                   axis_sizes: Union[int, tuple[int, ...]],
                   expand: bool = False) -> torch.Tensor:
    if not isinstance(axes, tuple):
        axes = (axes,)
    if not isinstance(axis_sizes, tuple):
        axis_sizes = (axis_sizes,)

    if len(axes) != len(axis_sizes):
        raise ValueError('The number of axes and axis sizes must be equal.')

    shape = list(input.shape)

    if expand:
        for axis, size in sorted(zip(axes, axis_sizes), key=lambda x: axes[0], reverse=True):
            input = input.unsqueeze(axis)
            shape.insert(axis, size)
    else:
        for axis, size in zip(axes, axis_sizes):
            shape[axis] = size

    return input.broadcast_to(shape)


_TModule = TypeVar('_TModule', bound=torch.nn.Module)


def copy_module(module: _TModule,
                device: TorchDevice = None,
                dtype: TorchDType = None,
                non_blocking: bool = False) -> _TModule:
    from copy import deepcopy

    module = deepcopy(module)

    if device is not None or dtype is not None:
        module.to(device=device, dtype=dtype, non_blocking=non_blocking)

    return module

