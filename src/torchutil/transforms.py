import torch
import torch.nn as nn


class NormalizeMinMax(nn.Module):
    def __init__(self, epsilon: float = 1e-7, start_dim: int = 2):
        super().__init__()

        self.epsilon = epsilon
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.start_dim

        if x.ndim < dim:
            raise ValueError(f'Expected tensor with order of at least {dim} Got tensor of order {x.ndim} instead.')

        x_flat = x.flatten(start_dim=dim)
        minimum = torch.min(x_flat, dim=-1).values.reshape(x.shape[0], x.shape[1], *(1 for _ in range(x.ndim - dim)))
        maximum = torch.max(x_flat, dim=-1).values.reshape(x.shape[0], x.shape[1], *(1 for _ in range(x.ndim - dim)))

        epsilon = torch.tensor(self.epsilon, dtype=x.dtype, device=x.device)

        return (x - minimum) / torch.maximum(epsilon, maximum - minimum)
