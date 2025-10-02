import os
from enum import Enum
from typing import Optional, Union, cast, Callable, Literal, NamedTuple, Sequence

import PIL.Image
import numpy as np
import torch

import torchvision.transforms.v2 as tt

import torchutil.transforms
import segment_anything
from segment_anything.modeling.sam import Sam

from pretrained import FeatureBackboneInvoker, FeatureBackboneFactory
from torchutil import TorchDevice


class MedSAMModels(Enum):
    vit_b = ('vit_b',
             '1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_',
             'medsam_vit_b.pth')

    @classmethod
    def _missing_(cls, value):
        for v in cls:
            if v.value[0] == value:
                return v

    @property
    def image_embedding_feature_count(self) -> int:
        if self == MedSAMModels.vit_b:
            return 256
        else:
            raise RuntimeError('Unsupported mode.')

    @property
    def image_embedding_resolution(self) -> int:
        if self == MedSAMModels.vit_b:
            return 64
        else:
            raise RuntimeError('Unsupported mode.')

    @property
    def input_resolution(self) -> int:
        if self == MedSAMModels.vit_b:
            return 1024
        else:
            raise RuntimeError('Unsupported mode.')


class SamModel(FeatureBackboneFactory):

    def __init__(self, description: MedSAMModels, model: Sam) -> None:
        self._description = description
        self._sam = model

    @property
    def model(self) -> Sam:
        return self._sam

    @property
    def description(self) -> MedSAMModels:
        return self._description

    def create_invoker(
            self,
            transform: Union[bool, Callable[[Union[torch.Tensor, PIL.Image.Image, np.ndarray]], torch.Tensor]] = True) \
            -> 'SamInvoker':
        return SamInvoker(self._description, self.model, transform)

    def to(self, device: TorchDevice) -> None:
        self.model.to(device=device)

    @property
    def feature_count(self) -> int:
        return self.description.image_embedding_feature_count

    @property
    def input_count(self) -> int:
        return 3


def medsam(model_dir: str,
           model: Union[MedSAMModels, str, Literal['vit_b']] = 'vit_b',
           load_checkpoint: bool = True,
           download: bool = True,
           device: TorchDevice = None) -> SamModel:
    model = MedSAMModels(model)

    checkpoint_path = None

    if load_checkpoint:
        model_id, src_url, dst_file = model.value

        checkpoint_path = os.path.join(model_dir, dst_file)

        if not os.path.exists(checkpoint_path):
            if not download:
                raise FileNotFoundError(checkpoint_path)

            import requests

            # taken from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
            URL = "https://drive.usercontent.google.com/download"

            session = requests.Session()

            response = session.get(URL, params={'id': src_url, 'confirm': 't', 'export': 'download'}, stream=True)
            from util import first_where
            token = first_where(response.cookies.items(), lambda item: item[0].startswith('download_warning'),
                          default=(None, None))[1]

            if token:
                params = {'id': id, 'confirm': token}
                response.close()
                response = session.get(URL, params=params, stream=True)

            with open(checkpoint_path, 'wb') as stream:
                for chunk in response.iter_content(65536):
                    stream.write(chunk)

            response.close()

    sam = segment_anything.build_sam_vit_b(checkpoint_path)

    if device is not None:
        sam.to(device=device)

    sam.eval()

    return SamModel(model, sam)


def transform_demo() -> tt.Transform:
    return tt.Compose([
        tt.ToDtype(torch.get_default_dtype(), scale=True),
        tt.Resize((1024, 1024), tt.InterpolationMode.BICUBIC),
        torchutil.transforms.NormalizeMinMax(),
    ])


class BoundingRect(NamedTuple):
    xs: torch.Tensor
    ys: torch.Tensor
    widths: torch.Tensor
    heights: torch.Tensor


# in the following we encapsulate the steps found in
# https://github.com/bowang-lab/MedSAM/blob/7c5dfff6785a18aa8e5c1d2008e3edcf5bc0c4d0/utils/demo.py#L168

class _SamMaskDecoder:
    def __init__(self,
                 sam: Sam,
                 orig_sizes: Union[list[tuple[int, int]], tuple[int, int]],
                 image_embeddings: torch.Tensor,
                 sparse_prompt: torch.Tensor,
                 dense_prompt: torch.Tensor) -> None:
        self._sam = sam
        self._image_embeddings = image_embeddings
        self._sparse_prompt = sparse_prompt
        self._dense_prompt = dense_prompt
        self._orig_sizes = orig_sizes

    def decode_mask(self,
                    scaling_mode: Union[None, Literal[
                        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact']] = 'bilinear',
                    antialias: bool = False,
                    as_logits: bool = False) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        low_res_logits, _ = self._sam.mask_decoder(image_embeddings=self._image_embeddings,
                                                   image_pe=self._sam.prompt_encoder.get_dense_pe(),
                                                   sparse_prompt_embeddings=self._sparse_prompt,
                                                   dense_prompt_embeddings=self._dense_prompt,
                                                   multimask_output=True)

        if as_logits:
            pred = low_res_logits
        else:
            pred = torch.sigmoid(low_res_logits)

        if scaling_mode is not None:
            scaling_params = dict(mode=scaling_mode, antialias=antialias, align_corners=False)
            if isinstance(self._orig_sizes, list):
                return [torch.nn.functional.interpolate(p, sz, **scaling_params)
                        for p, sz in zip(pred, self._orig_sizes)]
            else:
                return torch.nn.functional.interpolate(pred, self._orig_sizes, **scaling_params)

        return pred


class _SamPrompter:
    def __init__(self,
                 sam: Sam,
                 images_shape: tuple[int, int, int, int],
                 orig_sizes: Union[list[tuple[int, int]], tuple[int, int]],
                 image_embeddings: torch.Tensor) -> None:
        self._sam = sam
        self._image_embeddings = image_embeddings
        self._images_shape = images_shape
        self._orig_sizes = orig_sizes

    @property
    def image_embeddings(self) -> torch.Tensor:
        return self._image_embeddings

    def with_bounding_boxes(self, bounding_boxes: Optional[BoundingRect] = None) -> _SamMaskDecoder:
        N, C, H, W = self._images_shape
        device = self._image_embeddings.device

        if bounding_boxes is None:

            dtype = torch.get_default_dtype()
            zero = torch.tensor(N, device=device, dtype=dtype)
            widths = torch.full((N,), W, device=device, dtype=dtype)
            heights = torch.full((N,), H, device=device, dtype=dtype)

            bounding_boxes = BoundingRect(zero, zero, widths, heights)
        else:
            shape = None
            for t in bounding_boxes:
                if t.ndim not in (1,):
                    raise ValueError('The coordinates and sizes must be of order 1.')
                if len(t) != N:
                    raise ValueError('All tensors in `bounding_boxes` must have the same length as the images.')
                if t.device != device:
                    raise ValueError('The coordinates and sizes must be on the same device as the embeddings.')

                if shape is None:
                    shape = t.shape
                elif shape != t.shape:
                    raise ValueError('All coordinates and sizes must have the same shape.')

        if (bounding_boxes.widths < 0).any() or (bounding_boxes.heights < 0).any():
            raise ValueError('All widths and heights must be non-negative.')

        # adjust bounding boxes to 1024 and stack
        h_scale, v_scale = 1024 / W, 1024 / H

        bb = torch.stack([bounding_boxes.xs * h_scale,
                          bounding_boxes.ys * v_scale,
                          (bounding_boxes.widths + bounding_boxes.xs) * h_scale,
                          (bounding_boxes.heights + bounding_boxes.ys) * v_scale], dim=-1)

        # `bb` is a tensor of shape Nx4, change it to Nx1x4
        bb = bb.unsqueeze(1)

        sparse_prompt, dense_prompt = self._sam.prompt_encoder(points=None, boxes=bb, masks=None)

        return _SamMaskDecoder(self._sam, self._orig_sizes, self._image_embeddings, sparse_prompt, dense_prompt)


class SamInvoker(FeatureBackboneInvoker):
    def __init__(
            self,
            description: MedSAMModels,
            sam: Sam,
            transform: Union[bool, Callable[[Union[torch.Tensor, PIL.Image.Image, np.ndarray]], torch.Tensor]] = True) \
            -> None:
        if isinstance(transform, bool):
            transform = transform_demo() if transform else None

        self._description = description
        self._transform = transform
        self._sam = sam

        from torchutil import copy_module

        self._transform_meta = copy_module(transform,
                                           'meta') if isinstance(transform, torch.nn.Module) else transform

    @property
    def model(self) -> Sam:
        return self._sam

    @property
    def description(self) -> MedSAMModels:
        return self._description

    @property
    def identifier(self) -> str:
        return 'medsam'

    @property
    def name(self) -> str:
        return 'MedSAM'
    
    @property
    def device(self) -> torch.device:
        return self.model.device

    def embed(self, images: Union[torch.Tensor, Sequence[Union[torch.Tensor, PIL.Image.Image, np.ndarray]]]) \
            -> _SamPrompter:
        transform = self._transform

        if isinstance(images, torch.Tensor):
            orig_sz = cast(tuple[int, int], tuple(images.shape[2:]))

            if images.ndim != 4:
                from util.shapes import shape_to_string
                raise ValueError(f'Expected images of shape NxCxHxW, but got shape {shape_to_string(images.shape)}.')

            C = images.shape[1]

            if C not in (1, 3):
                raise ValueError(f'Expected 1 or 3 channels, but got {C}.')

            if C == 1:
                from torchutil import broadcast_axes
                images = broadcast_axes(images, 1, 3)
        else:
            orig_sz = []

            for img in images:
                if isinstance(img, torch.Tensor):
                    if img.ndim != 3:
                        raise ValueError('Expected images of shape CxHxW.')
                    H, W = img.shape[-2:]
                elif isinstance(img, np.ndarray):
                    if img.ndim != 3:
                        raise ValueError('Expected images of shape CxHxW.')
                    H, W = img.shape[-2:]
                elif isinstance(img, PIL.Image.Image):
                    H, W = img.height, img.width
                else:
                    from util import typename_of
                    raise TypeError(f'Unsupported type: {typename_of(img)}.')

                orig_sz.append((H, W))

        if transform is None:
            transform = tt.ToDtype(torch.get_default_dtype(), scale=True)

        from torchutil.seq import invoke_on_sequence

        transformed: torch.Tensor = invoke_on_sequence(images, transform, raise_if_not_same_size=True)

        if transformed.shape[-2:] != (1024, 1024):
            raise ValueError(
                f'After transformation, the input image must have size 1024x1024. Got {tuple(transformed.shape[-2:])}.')

        if transformed.shape[-3] != 3:
            raise ValueError(
                f'After transformation, the input image must have 3 channels. Got {transformed.shape[-3]}.')

        embeddings = self._sam.image_encoder(transformed)

        return _SamPrompter(self._sam, cast(tuple[int, int, int, int], tuple(transformed.shape)), orig_sz, embeddings)

    def features(self, images: Union[torch.Tensor, Sequence[Union[torch.Tensor, PIL.Image.Image, np.ndarray]]]) \
            -> torch.Tensor:
        return self.embed(images).image_embeddings

    def invoker(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.features

    @property
    def feature_count(self) -> int:
        return self.description.image_embedding_feature_count

    @property
    def input_count(self) -> int:
        return 3

    def feature_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError(f'Expected input of order 4, but got order {len(input_shape)}.')

        N, C, H, W = input_shape
        descr = self.description

        if self._transform_meta is not None:
            N, F, H, W = self._transform_meta(torch.empty((N, C, H, W), device='meta')).shape

        if not (descr.input_resolution == W == H):
            raise ValueError(f'The input image must have shape {H}, {W}. Got resolution {descr.input_resolution}.')

        if C not in (1, 3):
            raise ValueError('Expected the input to have 1 or 3 channels on axis 1.')

        output_res = descr.image_embedding_resolution

        return N, self.description.image_embedding_feature_count, output_res, output_res

    def to(self, device: TorchDevice) -> None:
        self._sam.to(device=device)
