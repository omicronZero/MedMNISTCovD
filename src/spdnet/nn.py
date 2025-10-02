# taken from https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/nn.py?ref_type=heads
from typing import Literal

import torch
import torch.nn as nn

from torchutil import TorchDTypeOpt, TorchDevice
from . import functional


class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """

    def __init__(self, ho: int, hi: int, ni: int, no: int, dtype: TorchDTypeOpt = None, device: TorchDevice = None):
        super(BiMap, self).__init__()

        self._W = functional.StiefelParameter(torch.empty(ho, hi, ni, no, dtype=dtype, device=device))
        self._ho = ho
        self._hi = hi
        self._ni = ni
        self._no = no

        functional.init_bimap_parameter(self._W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 4 or X.shape[-1] != X.shape[-2] or X.shape[-1] != self._ni or X.shape[-3] != self._hi:
            from util.shapes import shape_to_string
            raise ValueError(f'Expected a batch of shape NxCxQxQ with Q being {self._ni} and C being {self._hi}, but '
                             f'got shape {shape_to_string(X.shape)}.')

        return functional.bimap_channels(X, self._W)


class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        return functional.LogEig.apply(P)


class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        return functional.SqmEig.apply(P)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        return functional.ReEig.apply(P)


class BaryGeom(nn.Module):
    """
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.BaryGeom(x)


class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """

    def __init__(self, n: int, dtype: TorchDTypeOpt = None, device: TorchDevice = None):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = torch.eye(n, dtype=dtype, device=device)
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        self.weight = functional.SPDParameter(torch.eye(n, dtype=dtype, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N, h, n, n = X.shape

        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()

        if self.training:
            mean = functional.BaryGeom(X_batched)

            with torch.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)

            X_centered = functional.CongrG(X_batched, mean, 'neg')
        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, 'neg')

        X_normalized = functional.CongrG(X_centered, self.weight, 'pos')

        return X_normalized.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CovPool(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """

    def __init__(self, reg_mode: Literal['mle'] = 'mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool(f, self._reg_mode)


class CovPoolBlock(nn.Module):
    """
    Input f: L blocks of temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,L,n,T)
    Output X: L covariance matrices, shape (batch_size,L,1,n,n)
    """

    def __init__(self, reg_mode: Literal['mle'] = 'mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        ff = [functional.cov_pool(f[:, i, :, :], self._reg_mode)[:, None, :, :, :] for i in range(f.shape[1])]
        return torch.cat(ff, 1)


class CovPoolMean(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """

    def __init__(self, reg_mode: Literal['mle'] = 'mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool_mu(f, self._reg_mode)
