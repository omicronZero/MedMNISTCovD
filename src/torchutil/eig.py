import torch


def inveig(eigenvalue_diag_entries: torch.Tensor, eigenvectors: torch.Tensor, symmetrize: bool = True) -> torch.Tensor:
    m = eigenvectors * eigenvalue_diag_entries.unsqueeze(-2) @ eigenvectors.conj().transpose(-1, -2)

    if symmetrize:
        m = .5 * (m + m.transpose(-2, -1))

    return m
