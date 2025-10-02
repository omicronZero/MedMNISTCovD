from typing import Literal, Callable, Optional
import torch


def _as_weighted(loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 class_weights: torch.Tensor,
                 targets_to_float: bool,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean') \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError('`reduction` must be one of the following: \'mean\', \'sum\', or \'none\'.')

    class_weights = class_weights / class_weights.sum()

    def weighted_loss_function(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if targets_to_float:
            loss_target = target.float()
        else:
            loss_target = target

        L = loss_function(input, loss_target)

        if L.ndim == 0:
            raise RuntimeError('Unreduced loss expected. Set `reduction` on the `loss_function` supplied to '
                               '`_as_weighted` to `None`.')

        nonlocal class_weights

        class_weights = class_weights.to(device=L.device)

        L = L * class_weights[target]

        if reduction == 'mean':
            return L.mean()
        elif reduction == 'sum':
            return L.sum()
        else:
            return L

    return weighted_loss_function


def get_classifier_loss(
        loss_function: Literal[
            'bce', 'bce_logits', 'binary_focal', 'binary_focal_logits',
            'ce', 'ce_logits', 'categorical_focal', 'categorical_focal_logits',
            'auto_ce', 'auto_ce_logits', 'auto_focal', 'auto_focal_logits'],
        class_count: int,
        class_proportions: Optional[torch.Tensor] = None,
        weight_focal: bool = False) \
        -> tuple[Literal['binary', 'categorical'], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    if class_proportions is not None:
        if class_proportions.ndim != 1:
            raise ValueError(f'If indicated, `class_proportions` must be a tensor of order 1, but its order is '
                             f'{class_proportions.ndim}.')
        if len(class_proportions) != class_count:
            raise ValueError(
                f'`class_proportions` must have `class_count` items. Got `class_count=={class_count}`, but '
                f'`class_proportions` has length {len(class_proportions)}.')

    if loss_function in ('bce', 'bce_logits', 'binary_focal', 'binary_focal_logits'):
        if class_count != 2:
            raise ValueError(f'A binary classification loss has been indicated, but {class_count} classes were '
                             f'indicated.')
        # sticking to the PyTorch recommendations, we take the `pos_weight` as follows (see
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html):
        class_weights = None if class_proportions is None else class_proportions[0] / class_proportions[1]
    else:
        class_weights = None if class_proportions is None else 1 / class_proportions

    params = {}

    # map the auto-mode
    if loss_function == 'auto_ce':
        loss_function = 'bce' if class_count == 2 else 'ce'
    elif loss_function == 'auto_ce_logits':
        loss_function = 'bce_logits' if class_count == 2 else 'ce_logits'
    elif loss_function == 'auto_focal':
        loss_function = 'binary_focal' if class_count == 2 else 'categorical_focal'
    elif loss_function == 'auto_focal_logits':
        loss_function = 'binary_focal_logits' if class_count == 2 else 'categorical_focal_logits'

    # map the loss to the function we use
    if loss_function == 'bce':
        def binary_cross_entropy(
                input: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                size_average: Optional[bool] = None,
                reduce: Optional[bool] = None,
                reduction: str = "mean",
        ) -> torch.Tensor:
            target = target.float()

            return torch.nn.functional.binary_cross_entropy(input, target, weight, size_average, reduce, reduction)

        loss_function = binary_cross_entropy
        is_categorical = False
        is_focal = False
    elif loss_function == 'bce_logits':

        def binary_cross_entropy_with_logits(
                input: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                size_average: Optional[bool] = None,
                reduce: Optional[bool] = None,
                reduction: str = "mean",
                pos_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            target = target.float()

            return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight, size_average, reduce,
                                                                        reduction, pos_weight)

        loss_function = binary_cross_entropy_with_logits
        is_categorical = False
        is_focal = False
    elif loss_function == 'ce':
        loss_function = torch.nn.functional.nll_loss
        is_categorical = True
        is_focal = False
    elif loss_function == 'ce_logits':
        loss_function = torch.nn.functional.cross_entropy
        is_categorical = True
        is_focal = False
    elif loss_function in ('binary_focal', 'binary_focal_logits',
                           'categorical_focal', 'categorical_focal_logits'):
        raise NotImplementedError('Focal losses have been removed.')
    else:
        raise ValueError(f'Loss {repr(loss_function)} is not supported.')

    apply_weighting = class_proportions is not None and (weight_focal or not is_focal)

    if apply_weighting:
        params['reduction'] = 'none'

    if len(params) > 0:
        from functools import partial
        loss_function = partial(loss_function, **params)

    if apply_weighting:
        loss_function = _as_weighted(loss_function, class_weights,
                                     targets_to_float=not is_categorical)

    return 'categorical' if is_categorical else 'binary', loss_function
