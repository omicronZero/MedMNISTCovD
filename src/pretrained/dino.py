from enum import Enum
from typing import Literal, Union, Any, Optional, Sequence, Callable, TypedDict, cast

import PIL.Image
import numpy as np
import torch.nn
import torchvision.transforms.v2 as tt

from pretrained import FeatureBackboneInvoker, FeatureBackboneFactory
from torchutil import TorchDevice


class DINOv2Models(Enum):
    vit_s_14 = 'dinov2_vits14'
    vit_b_14 = 'dinov2_vitb14'
    vit_l_14 = 'dinov2_vitl14'
    vit_g_14 = 'dinov2_vitg14'

    @classmethod
    def _missing_(cls, value: Any) -> Optional['DINOv2Models']:
        if isinstance(value, str):
            value = value.upper()
            if value == 'S':
                return DINOv2Models.vit_s_14
            elif value == 'B':
                return DINOv2Models.vit_b_14
            elif value == 'L':
                return DINOv2Models.vit_l_14
            elif value == 'G':
                return DINOv2Models.vit_g_14

        return None

    @property
    def patch_size(self) -> int:
        return 14

    @property
    def feature_count(self) -> int:
        if self == DINOv2Models.vit_s_14:
            return 384
        elif self == DINOv2Models.vit_b_14:
            return 768
        elif self == DINOv2Models.vit_l_14:
            return 1024
        elif self == DINOv2Models.vit_g_14:
            return 1536
        else:
            raise RuntimeError('Unsupported model.')

    @property
    def heads(self) -> int:
        if self == DINOv2Models.vit_s_14:
            return 6
        elif self == DINOv2Models.vit_b_14:
            return 12
        elif self == DINOv2Models.vit_l_14:
            return 16
        elif self == DINOv2Models.vit_g_14:
            return 24
        else:
            raise RuntimeError('Unsupported model.')

    @property
    def feature_count_per_head(self) -> int:
        return self.feature_count // self.heads


class DINOv2FeatureOutputs(TypedDict):
    x_norm_clstoken: torch.Tensor
    x_norm_regtokens: torch.Tensor
    x_norm_patchtokens: torch.Tensor
    x_prenorm: torch.Tensor
    masks: Optional[torch.Tensor]


class DINOv2Invoker(FeatureBackboneInvoker):
    def __init__(self,
                 dino: 'DINOv2Model',
                 transform: Union[
                     None,
                     Literal['anna'],
                     Callable[[Union[torch.Tensor, np.ndarray, PIL.Image.Image]], torch.Tensor]]) -> None:
        if transform == 'anna':
            transform = dino_v2_transformation_anna()

        transform: Optional[Callable[[Union[torch.Tensor, np.ndarray, PIL.Image.Image]], torch.Tensor]]

        self._dino = dino
        self._transform: Optional[Callable[[Union[torch.Tensor, np.ndarray, PIL.Image.Image]], torch.Tensor]] \
            = transform

        from torchutil import copy_module

        self._transform_meta = copy_module(transform,
                                           'meta') if isinstance(transform, torch.nn.Module) else transform

    @property
    def device(self) -> torch.device:
        for p in self.model.model.parameters():
            return p.device
        raise RuntimeError()  # should never happen

    @property
    def model(self) -> 'DINOv2Model':
        return self._dino

    @property
    def description(self) -> DINOv2Models:
        return self._dino.description

    @property
    def distilled(self) -> bool:
        return self._dino.distilled

    @property
    def with_registers(self) -> bool:
        return self._dino.with_registers

    @property
    def feature_count(self) -> int:
        return self._dino.feature_count

    @property
    def input_count(self) -> int:
        return self._dino.input_count

    @property
    def transform(self) \
            -> Optional[Callable[
                [Union[torch.Tensor, np.ndarray, PIL.Image.Image], torch.Tensor], torch.Tensor]]:
        return self._transform

    def features(self,
                 images: Union[torch.Tensor, Sequence[Union[torch.Tensor, PIL.Image.Image, np.ndarray]]],
                 masks: Union[None, torch.Tensor, list[torch.Tensor]] = None,
                 unflatten: bool = False) \
            -> torch.Tensor:
        result = self._dino._features_core(images, masks, self._transform,
                                           ('x_norm_patchtokens',))['x_norm_patchtokens']

        assert result.ndim == 3

        if unflatten:
            import math
            F = round(math.sqrt(result.shape[1]))
            result = result.moveaxis(1, -1).reshape(result.shape[0], result.shape[2], F, F)

        return result

    def invoker(self) -> Callable[[torch.Tensor], torch.Tensor]:
        from functools import partial
        return partial(self.features, unflatten=True)

    def to(self, device: TorchDevice) -> None:
        self._dino.model.to(device=device)

    @property
    def identifier(self) -> str:
        return self._dino.identifier

    @property
    def name(self) -> str:
        parts = ['DinoV2']

        if self.description == DINOv2Models.vit_s_14:
            parts.append('ViT-Small')
        elif self.description == DINOv2Models.vit_b_14:
            parts.append('ViT-Base')
        elif self.description == DINOv2Models.vit_l_14:
            parts.append('ViT-Large')
        elif self.description == DINOv2Models.vit_g_14:
            parts.append('ViT-Giant')
        else:
            parts.append('Unknown model size')

        if self.with_registers:
            parts.append('with registers')

        if self.distilled:
            parts.append('distilled')

        return ' '.join(parts)

    def feature_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError(f'Expected input of order 4, but got order {len(input_shape)}.')

        N, C, H, W = input_shape
        descr = self.description

        N, C, H, W = self._transform_meta(torch.empty((N, C, H, W), device='meta')).shape

        if C not in (1, 3):
            raise ValueError('Expected the input to have 1 or 3 channels on axis 1.')

        h = H // descr.patch_size
        w = W // descr.patch_size

        if h * descr.patch_size != H:
            raise ValueError(f'Height must be a multiple of the current instance\'s patch size ({descr.patch_size}). '
                             f'Got {H}.')

        if w * descr.patch_size != W:
            raise ValueError(f'Width must be a multiple of the current instance\'s patch size ({descr.patch_size}). '
                             f'Got {w}.')

        return N, descr.feature_count, h, w


class DINOv2Model(FeatureBackboneFactory):
    def __init__(self,
                 description: DINOv2Models,
                 distilled: bool,
                 with_registers: bool,
                 model: torch.nn.Module,
                 identifier: str) -> None:

        self._description = description
        self._distilled = distilled
        self._with_registers = with_registers
        self._model = model
        self._identifier = identifier

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def description(self) -> DINOv2Models:
        return self._description

    @property
    def distilled(self) -> bool:
        return self._distilled

    @property
    def with_registers(self) -> bool:
        return self._with_registers

    @property
    def input_count(self) -> int:
        return 3

    @property
    def feature_count(self) -> int:
        return self.description.feature_count

    def features_detailed(self,
                          images: Union[torch.Tensor, Sequence[Union[torch.Tensor, PIL.Image.Image, np.ndarray]]],
                          masks: Union[None, torch.Tensor, list[torch.Tensor]] = None) \
            -> DINOv2FeatureOutputs:
        return DINOv2FeatureOutputs(**self._features_core(images, masks))

    def _features_core(self,
                       images: Union[torch.Tensor, Sequence[Union[torch.Tensor, PIL.Image.Image, np.ndarray]]],
                       masks: Union[None, torch.Tensor, list[torch.Tensor]] = None,
                       transform: Union[None, Callable[[torch.Tensor], torch.Tensor], Literal['anna']] = None,
                       keep: tuple[str, ...] = ()) -> dict[str, Optional[torch.Tensor]]:
        if transform == 'anna':
            transform = dino_v2_transformation_anna()
        transform: Optional[torch.nn.Module]

        # see https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L254

        if transform is None and not isinstance(images, torch.Tensor):
            transform = tt.ToDtype(torch.get_default_dtype(), scale=True)

        if transform is not None:
            from torchutil.seq import invoke_on_sequence
            images = invoke_on_sequence(images, transform)

        # to RGB
        if isinstance(images, torch.Tensor):
            if images.shape[1] == 1:
                from torchutil import broadcast_axes
                images = broadcast_axes(images, 1, 3)
        else:
            from torchutil import broadcast_axes
            images = [broadcast_axes(img, 0, 3) for img in images]

        result = {}

        for i, img in enumerate(images):
            img: torch.Tensor
            mask = None if masks is None else masks[i]

            with torch.no_grad():
                dct = self._model.forward_features(img.unsqueeze(0), None if mask is None else mask.unsqueeze(0))

            for k, v in dct.items():
                if keep != () and k not in keep:
                    # frees unnecessary data
                    continue

                r = result.get(k)

                if r is None:
                    result[k] = r = []

                r.append(v)

        return {k: torch.cat(v) for k, v in result.items()}

    def create_invoker(
            self,
            transform: Union[
                None,
                bool,
                Literal['anna'],
                Callable[[Union[torch.Tensor, np.ndarray, PIL.Image.Image]], torch.Tensor]] = True) \
            -> DINOv2Invoker:
        if isinstance(transform, bool):
            if transform:
                transform = 'anna'
            else:
                transform = None

        return DINOv2Invoker(self, cast(Union[
                                            None,
                                            Literal['anna'],
                                            Callable[[Union[torch.Tensor, np.ndarray, PIL.Image.Image]], torch.Tensor]],
                                        transform))


def dino_v2(model_dir: str,
            backbone: Union[DINOv2Models, Literal['S', 'B', 'L', 'G', 's', 'b', 'l', 'g'], str],
            with_registers: bool = False,
            distilled: bool = False,
            device: TorchDevice = None,
            download: bool = True,
            ) -> DINOv2Model:
    model_descr = DINOv2Models(backbone)
    backbone = model_descr.value

    if with_registers:
        backbone = backbone + '_reg'

    if distilled:
        backbone = backbone + '_lc'

    from pretrained import load_hub_model

    optkwargs = {}

    if not download:
        optkwargs['source'] = 'local'

    model = load_hub_model(model_dir,
                           'facebookresearch/dinov2',
                           backbone,
                           device=device,
                           **optkwargs)

    model.eval()

    return DINOv2Model(model_descr, distilled, with_registers, model, backbone)


def dino_v2_transformation_anna(size: int = 520, margin: int = 1) -> tt.Transform:
    return tt.Compose([
        tt.Resize(size),
        tt.CenterCrop(size - 2 * margin),
        tt.ToTensor(),
        tt.Normalize(mean=(.5,), std=(.2,))
    ])
