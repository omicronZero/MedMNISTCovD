from typing import TypeVar, Sequence, Callable, Union, overload, Literal, Iterable

import torch

_T = TypeVar('_T')


@overload
def invoke_on_sequence(inputs: Sequence[_T],
                       evaluator: Callable[[_T], torch.Tensor],
                       raise_if_not_same_size: bool = False,
                       stack_if_same_size: bool = True) \
        -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    ...


@overload
def invoke_on_sequence(inputs: Sequence[_T],
                       evaluator: Callable[[_T], torch.Tensor],
                       raise_if_not_same_size: Literal[True],
                       stack_if_same_size: Literal[True] = True) \
        -> torch.Tensor:
    ...


def invoke_on_sequence(inputs: Iterable[_T],
                       evaluator: Callable[[_T], torch.Tensor],
                       raise_if_not_same_size: bool = False,
                       stack_if_same_size: bool = True) \
        -> Sequence[torch.Tensor]:
    if isinstance(inputs, torch.Tensor):
        outp = evaluator(inputs)

        if len(outp) != len(inputs):
            raise ValueError(
                f'`evaluator` must preserve the size of the first axis. Expected {len(inputs)}, but got {len(outp)}.')

        return outp

    same_size = True
    size = None
    result = []

    for inp in inputs:
        outp = evaluator(inp.unsqueeze(0))

        if outp.shape[0] != 1:
            raise ValueError('`evaluator` must preserve the size of the first axis.')

        outp = outp.squeeze(0)

        if size is None:
            size = outp.shape
        elif size != outp.shape:
            if raise_if_not_same_size:
                raise ValueError(
                    f'`evaluator` must preserve the size of the first axis. Expected {size}, but got {len(outp)}.')
            same_size = False

        result.append(outp)

    if len(result) == 0:
        raise ValueError('`tensors` must not be empty.')

    if same_size and stack_if_same_size:
        result = torch.stack(result)

    return result
