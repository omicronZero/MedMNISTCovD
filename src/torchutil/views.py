import torch
from typing import Sequence, Union


def sliding_window(tensor: torch.Tensor,
                   *window_sizes_strides: Union[int, tuple[int, int]],
                   axes: Union[int, Sequence[int]] = 0) -> torch.Tensor:
    if not isinstance(axes, Sequence):
        axes = range(axes, axes + len(window_sizes_strides))

    if len(axes) != len(window_sizes_strides):
        raise ValueError('If axes are specified their length must be equal to the number of window sizes.')

    result = tensor

    from util import abs_index

    axes = [abs_index(axis, tensor.ndim) for axis in axes]

    # we go through the sorted axes-window_size_strides-pairs from low to high and expand the respective axis
    axis_window_size_pairs = sorted(zip(axes, window_sizes_strides), key=lambda x: x[0])

    for axis, window_size in axis_window_size_pairs:
        stride = 1

        if isinstance(window_size, tuple):
            window_size, stride = window_size

        result = result.unfold(axis, window_size, stride)

    return result
