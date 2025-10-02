from typing import Optional, Callable, Any, Union, Literal

import torch

import torchutil
import lightning.pytorch as pl

from nn_lit import ClassifierModel


class SPDNetClassifierModel(ClassifierModel):
    def __init__(self,
                 result_dir: str,
                 model_identifier: str,
                 model: torch.nn.Module,
                 labels: tuple[str, ...],
                 loss_function: Union[
                     torch.nn.Module,
                     Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     Literal[
                         'bce', 'bce_logits',
                         'ce', 'ce_logits',
                         'auto_ce', 'auto_ce_logits']
                 ] = 'auto_ce_logits',
                 class_proportions: Optional[torch.Tensor] = None,
                 trainer: Optional[pl.Trainer] = None,
                 training_batch_size: int = 32,
                 val_test_batch_size: int = 256,
                 trainer_kwargs: Optional[dict[str, Any]] = None,
                 optimizer_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(result_dir, model_identifier, labels, trainer, loss_function, class_proportions,
                         training_batch_size, val_test_batch_size, trainer_kwargs)

        self._model = model
        self._optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    def _create_optimizer(self) -> torch.optim.Optimizer:
        from spdnet.optimizers import MixOptimizer

        return MixOptimizer(self.parameters(), **self._optimizer_kwargs)

    def _forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self._model(*input)


class CovarianceLayer(torch.nn.Module):
    def __init__(self, ndim: int) -> None:
        if ndim <= 0:
            raise ValueError('`ndim` must be positive.')

        self._ndim = ndim

        super().__init__()

    def forward(self,
                images: torch.Tensor,
                weights: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ndim = self._ndim

        if images.ndim < ndim + 1:
            raise ValueError(f'`images` must be a tensor of an order of at least {ndim}.')

        from torchutil.bcov import bcov

        N, (C, *D) = images.shape[:-ndim - 1], images.shape[-ndim - 1:]

        if weights is not None:
            if weights.shape != (*N, *D):
                from util.shapes import shape_to_string
                raise ValueError(f'For a value for `images` of shape {shape_to_string(images.shape)}, `weights` must '
                                 f'be of shape {shape_to_string((*N, *D))}, but got '
                                 f'{shape_to_string(weights.shape)}.')

        if mask is not None:
            if mask.shape != (*N, *D):
                from util.shapes import shape_to_string
                raise ValueError(f'For a value for `images` of shape {shape_to_string(images.shape)}, `weights` must '
                                 f'be of shape {shape_to_string((*N, *D))}, but got {shape_to_string(mask.shape)}.')

        images = images.flatten(start_dim=-ndim)
        weights = weights if weights is None else weights.flatten(start_dim=-ndim)
        mask = None if mask is None else mask.flatten(start_dim=-ndim)

        return bcov(images, weights, mask)


class SPDNet(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 *spd_dims: int,
                 batch_norm: bool = False,
                 dtype: torchutil.TorchDTypeOpt = None,
                 device: torchutil.TorchDevice = None) -> None:
        super().__init__()

        import spdnet.nn as spdnet

        layers = {}

        reeig = spdnet.ReEig()

        inp_dim = input_size

        for i, outp_dim in enumerate(spd_dims, start=1):
            layers[f'bimap{i}'] = spdnet.BiMap(1, 1, inp_dim, outp_dim, dtype=dtype, device=device)

            if batch_norm:
                layers[f'batchnorm{i}'] = spdnet.BatchNormSPD(outp_dim, dtype=dtype, device=device)

            layers[f'reeig{i}'] = reeig

            inp_dim = outp_dim

        layers['logeig'] = spdnet.LogEig()

        from collections import OrderedDict

        self._input_dim = input_size
        self.spd_transform = torch.nn.Sequential(OrderedDict(layers))
        self.linear = torch.nn.Linear(inp_dim ** 2, output_size, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError('Expected `x` to be of an order of at least 3.')

        *B, Q, Qp = x.shape

        if Q != Qp:
            from util.shapes import shape_to_string
            raise ValueError(f'The last two axes of `x` must have the same size. Got shape {shape_to_string(x.shape)}.')

        if Q != self._input_dim:
            from util.shapes import shape_to_string
            raise ValueError(
                f'Expected shape ...xQxQ with Q being {self._input_dim}. Got shape {shape_to_string(x.shape)}.')

        # the SPDNet-implementation does not support arbitrarily shaped batches --> flatten
        x = x.flatten(end_dim=-3)

        B_flat, Qp, Qpp = x.shape

        assert Qp == Qpp

        # for the SPDNet-channels, we unsqueeze the first axis (channels were introduced in a subsequent paper which
        # was based on the original work. The used implementation implements these channels, therefore, we just set this
        # to 1 by unsqueezing)
        x = x.unsqueeze(1)

        x: torch.Tensor = self.spd_transform(x)

        assert x.ndim == 4

        F = x.shape[-3]

        assert x.shape == (B_flat, F, x.shape[-1], x.shape[-1])
        assert F == 1

        # now, let's undo the batching
        Qp = x.shape[-1]
        x = x.reshape(*B, Qp, Qp)

        x_vec = x.flatten(start_dim=-2)

        y = self.linear(x_vec)

        return y


class BasicSPDNet(torch.nn.Module):

    def __init__(self, input_dim: int, output_size: int, batch_norm: bool = False,
                 device: torchutil.TorchDevice = None, dtype: torchutil.TorchDTypeOpt = None) -> None:
        super().__init__()

        dim = input_dim
        dim1 = (dim + 1) // 2
        dim2 = (dim1 + 1) // 2
        dim3 = (dim2 + 1) // 2

        self._spdnet = SPDNet(dim, output_size, dim1, dim2, dim3, batch_norm=batch_norm,
                              device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._spdnet(input)
