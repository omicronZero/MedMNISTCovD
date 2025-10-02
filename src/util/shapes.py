from typing import Union, Callable


def adjust_axis(shape: tuple[int, ...],
                axis: Union[int, tuple[int, ...]],
                new_value: Union[None, int, Callable[[int], int], tuple[int, Callable[[int], int], ...]])\
        -> tuple[int, ...]:
    new_shape = list(shape)

    if isinstance(axis, tuple):
        if isinstance(new_value, tuple):
            if len(axis) != len(new_value):
                raise ValueError('If axis and new_value are tuples they must have the same length.')

            for a, v in zip(axis, new_value):
                new_shape[a] = v(new_shape[a]) if callable(v) else v
        else:
            raise TypeError('axis and new_values must both be tuples if one of them is.')
    elif isinstance(new_value, tuple):
        raise TypeError('axis and new_values must both be tuples if one of them is.')
    else:
        new_shape[axis] = new_value(new_shape[axis]) if callable(new_value) else new_value

    return tuple(s for s in new_shape if s is not None)


def move_axis_on_shape(shape: tuple[int, ...],
                       source_axis: int,
                       target_axis: int) -> tuple[int, ...]:
    shape = list(shape)
    fa = shape.pop(source_axis)

    shape.insert(target_axis, fa)

    return tuple(shape)


def shape_to_string(shape: tuple[Union[str, int], ...], x: bool = False) -> str:
    if len(shape) == 0:
        return '()'
    elif len(shape) == 1:
        return f'({shape[0]},)'

    if x:
        return 'x'.join(map(str, shape))
    else:
        return f'({", ".join(map(str, shape))})'


def shape_to_x(shape: tuple[Union[str, int], ...]) -> str:
    return shape_to_string(shape, x=True)
