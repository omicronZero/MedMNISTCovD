from typing import Optional, runtime_checkable, Protocol, Callable, Union, TypeVar, Any, cast, Iterable

_T = TypeVar('_T')
_TOpt = TypeVar('_TOpt')


def try_abs_index(index: int, length: int) -> Optional[int]:
    if length < 0:
        raise ValueError('Non-negative length expected.')
    if index < 0:
        index += length

    if index < 0 or index >= length:
        return None

    return index


def abs_index(index: int, length: int, name: Optional[str] = None) -> int:
    idx = try_abs_index(index, length)

    if idx is None:
        if name is None:
            name = 'Index'
        elif not name.startswith('`') and not name.startswith('´'):
            name = f'`{name}`'

        raise IndexError(name + ' does not fall into the collection.')

    return idx


def abs_range(start: int, end: int, length: int, end_exclusive: bool = True) -> tuple[int, int]:
    if start == end and end_exclusive:
        # empty ranges are always allowed
        return 0, 0

    start = abs_index(start, length, '`start`')
    if end < 0:
        end += length

    if end >= 0:
        if start > end:
            raise ValueError('`start` must be less than or equal to `end`.')

        if end_exclusive:
            if end <= length:
                return start, end
        elif end < length:
            return start, end

    raise IndexError('`end` does not fall into the collection.')


def coalesce(value: Optional[_T], default: _TOpt) -> Union[_T, _TOpt]:
    if value is None:
        return default
    return value


@runtime_checkable
class Named(Protocol):
    __name__: str


@runtime_checkable
class ModuleNamedObject(Protocol):
    __name__: str
    __module__: str


@runtime_checkable
class QualifiedNamedObject(Protocol):
    __module__: str
    __qualname__: str


def nameof(named: Union[QualifiedNamedObject, ModuleNamedObject, Named, Callable, property],
           include_module: bool = True,
           prune_builtins: bool = True) -> str:
    if isinstance(named, property):
        f = named.fget

        if f is None:
            f = named.fset
        if f is None:
            f = named.fdel

        if f is None:
            raise RuntimeError('Property does not define a getter, setter, or deleter.')

        named = f

    if isinstance(named, QualifiedNamedObject):
        module = named.__module__
        name = named.__qualname__
    elif isinstance(named, ModuleNamedObject):
        module = named.__module__
        name = named.__name__
    elif isinstance(named, Named):
        name = named.__name__
        module = None
    else:
        name = getattr(named, '__name__', None)
        name = coalesce(getattr(named, '__qualname__', name), name)
        if name is None:
            try:
                tn = f' Got object of type {typename_of(named)}.'
            except TypeError:
                tn = ''

            raise TypeError(f'The indicated type does not provide a name. Expected type `{nameof(Named)}`, '
                            f'`{nameof(ModuleNamedObject)}`, or `{nameof(QualifiedNamedObject)}`. {tn}')

        module = getattr(named, '__module__', None)

    if name is None:
        raise RuntimeError('`__name__` must not be `None`.')

    if not include_module or prune_builtins and module == 'builtins':
        module = None

    if module is None:
        return name
    else:
        return module + '.' + name


def typename(tp: type, prune_builtins: bool = True) -> str:
    return nameof(tp, prune_builtins=prune_builtins)


def typename_of(instance: Any, prune_builtins: bool = True, keep_type: bool = False) -> str:
    if not keep_type or not isinstance(instance, type):
        tp = type(instance)
    else:
        tp = cast(type, instance)

    return typename(tp, prune_builtins)


def next_dir(root_dir: str,
             sub_pattern: Callable[[int], str],
             create: bool = True,
             init: Optional[str] = None,
             start: int = 1) -> str:
    import os

    def get_full(p: str) -> str:
        if os.path.isabs(p):
            return p

        return os.path.join(root_dir, p)

    file = None if init is None else get_full(init)
    i = start

    attempts = set()

    if file is not None:
        attempts.add(file)

    while file is None or os.path.exists(file):
        file = get_full(sub_pattern(i))

        if file in attempts:
            raise RuntimeError(f'The same file was tried multiple times ({repr(file)}).')

        i += 1

    if create:
        os.makedirs(file, exist_ok=True)

    return file


def first_where(values: Iterable[_T], predicate: Callable[[_T], bool], default: _TOpt = ...) -> Union[_T, _TOpt]:
    for v in values:
        if predicate(v):
            return v

    if default is ...:
        raise ValueError('No value satisfied the predicate.')

    return default
