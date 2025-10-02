from enum import Enum
from types import EllipsisType
from typing import Union, Optional, Any, Literal, cast, Callable

import numpy as np
import torch
from torch.serialization import MAP_LOCATION

from pretrained import FeatureBackboneInvoker, FeatureBackboneFactory

import torch.nn as nn

import torchvision.models as mdl

from torchutil import TorchDevice


class BackboneModel(Enum):
    resnet18 = 'resnet18'

    @property
    def display_name(self) -> str:
        if self == BackboneModel.resnet18:
            return 'ResNet-18'
        else:
            raise RuntimeError('Unsupported model.')


class Backbone(FeatureBackboneInvoker):

    def __init__(self,
                 backbone: BackboneModel,
                 model: nn.Module,
                 feature_backbone: Optional[nn.Module],
                 requires_rgb: bool,
                 input_size_wh: Optional[tuple[int, int]],
                 backbone_output_layer_name: str,
                 backbone_output_feature_count: int,
                 weights: Optional[mdl.WeightsEnum],
                 input_dtype: torch.dtype,
                 apply_weight_transform: bool = True,
                 meta_feature_backbone: Optional[nn.Module] = None) -> None:
        if meta_feature_backbone is None and feature_backbone is not None:
            import copy
            meta_feature_backbone = copy.deepcopy(feature_backbone)
            meta_feature_backbone.to(device='meta')

        self._backbone_output_feature_count = backbone_output_feature_count
        self._backbone = backbone
        self._input_size = input_size_wh
        self._feature_backbone = feature_backbone
        self._input_dtype = input_dtype
        self._model = model
        self._backbone_output_layer_name = backbone_output_layer_name
        self._weights = weights
        self._requires_rgb = requires_rgb
        self._apply_weight_transform = apply_weight_transform

        transform = None if weights is None else weights.transforms()

        from torchutil import copy_module

        transform_meta = copy_module(transform, 'meta') if transform is not None else None

        self._transform = transform
        self._transform_meta = transform_meta
        self._meta_feature_backbone = meta_feature_backbone

    @property
    def device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device

    @property
    def backbone(self) -> BackboneModel:
        return self._backbone

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def backbone_output_layer_name(self) -> str:
        return self._backbone_output_layer_name

    @property
    def input_count(self) -> int:
        return 3 if self._requires_rgb else 1

    @property
    def input_size_wh(self) -> Optional[tuple[int, int]]:
        return self._input_size

    @property
    def identifier(self) -> str:
        return self._backbone.name

    @property
    def name(self) -> str:
        return self._backbone.display_name

    def invoker(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self._feature_backbone is None:
            raise RuntimeError('The current model does not have a feature backbone.')

        def handle(img: torch.Tensor) -> torch.Tensor:
            if img.ndim != 4:
                raise ValueError('Expected tensor of shape NxCxHxW.')

            if self._requires_rgb:
                C = img.shape[1]

                if C == 1:
                    from torchutil import broadcast_axes

                    img = broadcast_axes(img, 1, 3)

            if self._apply_weight_transform and self._transform is not None:
                img = self._transform(img)

            return self._feature_backbone(img)

        return handle

    @property
    def feature_count(self) -> int:
        return self._backbone_output_feature_count

    def to(self, device: TorchDevice) -> None:
        self._model.to(device=device)

    @property
    def weights(self) -> Optional[mdl.WeightsEnum]:
        return self._weights

    def with_output(self,
                    output: nn.Module,
                    on_model_copy: Union[bool, Literal['deep']] = False) -> nn.Module:
        model = self.model

        if on_model_copy:
            import copy

            if on_model_copy == 'deep':
                model = copy.deepcopy(model)
            else:
                model = copy.copy(model)

        setattr(model, self.backbone_output_layer_name, output)

        return model

    @property
    def transforms(self) -> Any:
        return self.weights.transforms

    def feature_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError(f'Expected `input_shape` to be of order 4, but got order {len(input_shape)}.')

        input_channels = self.input_count
        input_size = self.input_size_wh

        N, C, H, W = input_channels

        if input_channels != C:
            raise ValueError(f'Expected input to have {input_channels} channels, but got {C}.')

        if input_size is not None and input_size != (W, H):
            raise ValueError(f'Expected input to have size ({input_size}), but got ({W}, {H}).')

        meta_tensor = torch.empty(N, C, H, W, device='meta', dtype=self._input_dtype)

        if self._apply_weight_transform and self._transform_meta is not None:
            meta_tensor = self._transform_meta(meta_tensor)

        meta_tensor = self._meta_feature_backbone(meta_tensor)

        return tuple(meta_tensor.shape)


def backbone(model: Union[str, BackboneModel],
             class_count: Union[int, Literal['explicit2']],
             weight_source: Optional[str] = 'DEFAULT',
             freeze: bool = False,
             model_dir: Union[str, None, EllipsisType] = None,
             map_location: MAP_LOCATION = None,
             progress: bool = True,
             check_hash: bool = False,
             file_name: Optional[str] = None,
             weights_only: bool = False,
             apply_weight_transform: bool = True) \
        -> Backbone:
    if model_dir is ...:
        import project
        model_dir = project.get_models_dir('pretrained', 'torchvision')

    model = BackboneModel(model)

    input_size_wh: Optional[tuple[int, int]] = None
    input_dtype = torch.get_default_dtype()

    if model == BackboneModel.resnet18:
        model_inst = mdl.resnet18()
        weight_decl = mdl.ResNet18_Weights
        requires_rgb = True
        input_size_wh = 224, 224
    else:
        raise RuntimeError('Unsupported model.')

    wgts = None

    if weight_source is not None:
        if weight_source == 'DEFAULT':
            wgts = cast(mdl.WeightsEnum, getattr(weight_decl, 'DEFAULT'))
        else:
            wgts = weight_decl(weight_source)

    if wgts is not None:
        state_dict = wgts.get_state_dict(model_dir=model_dir,
                                         map_location=map_location,
                                         progress=progress,
                                         check_hash=check_hash,
                                         file_name=file_name,
                                         weights_only=weights_only)

        model_inst.load_state_dict(state_dict)

    if isinstance(model_inst, mdl.ResNet):
        backbone_layer_name = 'fc'
        backbone_output_size = model_inst.fc.in_features
        feature_inst = torch.nn.Sequential(model_inst.conv1,
                                           model_inst.bn1,
                                           model_inst.relu,
                                           model_inst.maxpool,
                                           model_inst.layer1,
                                           model_inst.layer2,
                                           model_inst.layer3,
                                           model_inst.layer4)
    else:
        raise RuntimeError('Implementation error.')

    if freeze:
        for param in model_inst.parameters(recurse=False):
            param.requires_grad = False

        for submodule_name, submodule in model_inst.named_children():
            if submodule_name == backbone_layer_name:
                continue

            for param in submodule.parameters(recurse=True):
                param.requires_grad = False

    if class_count is not None:
        if class_count == 2:
            # since we typically have binary classification with binary cross entropy and not softmax, set `class_count`
            # to 1
            class_count = 1
        elif class_count == 'explicit2':
            # the user can explicitly set to 2
            class_count = 2

        class_count: int

        setattr(model_inst, backbone_layer_name, nn.Linear(backbone_output_size, class_count))

    return Backbone(model, model_inst, feature_inst, requires_rgb, input_size_wh, backbone_layer_name,
                    backbone_output_size, wgts, input_dtype, apply_weight_transform=apply_weight_transform)
