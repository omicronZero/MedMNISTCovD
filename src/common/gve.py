from typing import Union, Literal, Iterable, Callable

import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA


class PretrainedFeatures:
    def __init__(self, model_dir: str, model: Union[str, Literal['dino', 'medsam']]) -> None:
        if model == 'dino':
            from pretrained.dino import dino_v2
            base = dino_v2(model_dir, 'L', download=True)
            feature_count = base.feature_count
            model = base.create_invoker()
        elif model == 'medsam':
            from pretrained.medsam import medsam
            base = medsam(model_dir, download=True)
            feature_count = base.feature_count
            model = base.create_invoker()
        else:
            raise RuntimeError(f'Unsupported model: {model}.')

        self._model = model
        self._feature_count = feature_count

    @property
    def feature_count(self) -> int:
        return self._feature_count

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim in (3, 4)

        squeeze = images.ndim == 3

        if squeeze:
            images = images.unsqueeze(0)

        N, C, H, W = images.shape

        with torch.no_grad():
            result = self._model.features(images.to(dtype=torch.float32))

        if result.ndim != 4:
            from math import sqrt
            h = w = round(sqrt(result.shape[-2]))

            result = result.reshape(N, h, w, -1).moveaxis(-1, -3)

        if squeeze:
            result = result.squeeze(0)

        return result


def image_to_pca(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = image.numpy(force=True)

    *N, C, H, W = image.shape

    return np.moveaxis(image, 0, -1).reshape((-1, C))


def image_from_pca(vec: np.ndarray, height: int, width: int) -> np.ndarray:
    D, C = vec.shape

    if D != width * height:
        raise ValueError('The combination of `width` and `height` does not match the vector\'s shape.')

    return np.moveaxis(vec.reshape((height, width, C)), -1, 0)


def pretrain_pca(imgs: Iterable[np.ndarray],
                 image_map: Callable[[torch.Tensor], torch.Tensor],
                 pca_dim: int = 16) -> IncrementalPCA:
    pca = IncrementalPCA(pca_dim)

    from common import tqdm
    for img in tqdm(imgs):
        feature_image = image_map(torch.tensor(img)).numpy(force=True)

        # move features to last axis, flatten remaining dimensions
        feature_vec = image_to_pca(feature_image)

        # fit incremental PCA to new data
        pca.partial_fit(feature_vec)

    return pca
