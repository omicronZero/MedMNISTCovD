from typing import Optional, Union

import torch


def bcov(x: torch.Tensor,
         weights: Optional[torch.Tensor] = None,
         mask: Optional[torch.Tensor] = None,
         correction: Union[int, bool, float] = 1.,
         use_weighted_mean: bool = True,
         divide_by_normalization_const: bool = True) \
        -> torch.Tensor:
    # desired shape: NxVxO where N is the batch size, V is the number of variables, O is the number of observations
    ndim = x.ndimension()
    if ((weights is not None and weights.ndimension() != ndim - 1)
            or (mask is not None and mask.ndimension() != ndim - 1)):
        raise ValueError('Weights and mask shape must be one less than that of x (allowed shapes: NxO or O).')

    if use_weighted_mean:
        # mask and weights do the same task if weighted_mean is True
        if weights is not None:
            if mask is not None:
                weights = weights * mask
        elif mask is not None:
            weights = mask
        mask = None

    if ndim == 1:
        # given shapes:
        # x: O
        # weights and mask: O
        x = x.unsqueeze(0).unsqueeze(0)

        if weights is not None:
            weights = weights.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)

        take = 0, 0
    elif ndim == 2:
        # given shapes:
        # x: VxO
        # weights and mask: O
        x = x.unsqueeze(0)

        if weights is not None:
            weights = weights.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)

        take = 0
    elif ndim == 3:
        # given shapes:
        # x: NxVxO
        # weights and mask: NxO
        take = None
        if weights is not None:
            weights = weights.unsqueeze(1)

        if mask is not None:
            mask = mask.unsqueeze(1)
    else:
        raise ValueError('Expected x of shape O, VxO, or NxVxO.')

    # x now has shape NxVxO
    # mask now has shape Nx1xO
    # weights now has shape Nx1xO

    if isinstance(correction, bool):
        correction = int(correction)

    O = x.shape[-1]

    # compute mean and move mask to weights, if necessary
    if mask is not None:
        # implies use_weighted_mean = False
        # we take the sum over the non-masked-out observations and divide by their count
        mask_counts = mask.sum(dim=-1, keepdim=True)
        feature_mean = (x * mask).sum(dim=-1, keepdim=True) / mask_counts
        if weights is None:
            weights = mask
            weighted_counts = mask_counts
        else:
            weights = weights * mask
            weighted_counts = weights.sum(dim=-1, keepdim=True)
    elif weights is not None:
        # if mask was not None, it is now assigned to weights or combined with it
        weighted_counts = weights.sum(dim=-1, keepdim=True)
        feature_mean = (x * weights).sum(dim=-1, keepdim=True) / weighted_counts
    else:
        weighted_counts = O
        feature_mean = x.mean(dim=-1, keepdim=True)

    # feature_mean has shape Nx1x1

    # this is Z = X - \mu where \mu is the mean or weighted mean
    centered = x - feature_mean

    if weights is not None:
        centered_weighted = weights * centered
    else:
        centered_weighted = centered

    # compute outer product C = Z @ Z^T = (X - \mu)(X - \mu)^T
    result = centered_weighted @ centered.transpose(-1, -2)

    # since result C has shape NxVxV, we can just divide by the Nx1x1 feature_mean (or by a singleton)
    if divide_by_normalization_const:
        if correction is None or correction == 0.:
            result = result / weighted_counts
        elif isinstance(weighted_counts, torch.Tensor):
            if weights.dtype == torch.bool:  # especially the case if we have a mask (squaring is redundant)
                result = result / (weighted_counts - correction)
            else:
                wsq_sum = weights.square().sum(dim=-1, keepdim=True)

                result = result / (weighted_counts - wsq_sum / weighted_counts * correction)
        else:
            result = result / (O + correction)

    # if we didn't receive a batch or just observations, take is set to a value. We take the first or first two axes
    if take is not None:
        result = result[take]

    return result
