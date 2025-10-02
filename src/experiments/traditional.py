from typing import Literal, Union, Callable, Optional

import torch

from common import log_euclidean_exp, descriptor_mean, log_euclidean_metric


def mdrm(train_covs: torch.Tensor, train_lbl: torch.Tensor, test_covs: torch.Tensor,
          remark: Optional[Callable[[str], None]] = None) -> torch.Tensor:
    assert train_lbl.ndim == 1 or train_lbl.ndim == 2 and train_lbl.shape[-1] == 1
    train_lbl = train_lbl.flatten()

    means = []
    print('  MDRM.')
    print('    Determining class centers...')

    for cls in train_lbl.unique():
        mask = train_lbl == cls

        class_cov = train_covs[mask]

        class_mean = descriptor_mean(class_cov)

        means.append(class_mean)

    means = torch.stack(means, dim=0)

    print('    Predicting...')
    distance_matrix = log_euclidean_metric(test_covs.unsqueeze(1), means.unsqueeze(0))
    prediction = distance_matrix.argmin(dim=-1)

    return prediction


def tslda(train_covs: torch.Tensor, train_lbl: torch.Tensor, test_covs: torch.Tensor,
          remark: Optional[Callable[[str], None]] = None,
          basepoint: Union[torch.Tensor, Literal['identity', 'mean']] = 'mean',
          threshold: float = 1e11) -> torch.Tensor:
    assert train_lbl.ndim == 1 or train_lbl.ndim == 2 and train_lbl.shape[-1] == 1
    train_lbl = train_lbl.flatten()

    *N1, Q11, Q12 = train_covs.shape
    *N2, Q21, Q22 = test_covs.shape

    assert len(train_covs) == len(train_lbl)
    assert Q11 == Q12 == Q21 == Q22
    assert threshold > 0.

    Q = Q11

    if isinstance(basepoint, str):
        print(f'  TSLDA at {basepoint}.')
        if basepoint == 'identity':
            basepoint = torch.eye(Q, device=train_covs.device, dtype=train_covs.dtype)
        elif basepoint == 'mean':
            basepoint = descriptor_mean(train_covs)

    if not isinstance(basepoint, torch.Tensor):
        raise ValueError('Invalid value indicated for `basepoint`.')

    print('    Mapping to tangent space...')
    train_ts = log_euclidean_exp(train_covs, basepoint)
    test_ts = log_euclidean_exp(test_covs, basepoint)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()

    train_mask = (train_ts < threshold).all(dim=-1)

    if remark is not None and (train_mask == False).any():
        remark(
            f'The following images exceeded the threshold of {threshold}: '
            f'{", ".join(map(str, (train_mask == False).argwhere().flatten().tolist()))}')

    print('    Fitting...')
    lda.fit(train_ts[train_mask].numpy(force=True), train_lbl[train_mask].numpy(force=True))

    print('    Predicting...')
    prediction_np = lda.predict(test_ts.numpy(force=True))

    return torch.tensor(prediction_np, dtype=train_lbl.dtype, device=train_lbl.device)