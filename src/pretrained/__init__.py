from abc import abstractmethod, ABC
from typing import Any, Literal, Union, Callable

import PIL.Image
import numpy as np
import torch

from torchutil import TorchDevice


def load_hub_model(
        model_dir: str,
        repo_or_dir: str,
        model: str,
        *args: Any,
        source: Literal['local', 'github'] = 'github',
        trust_repo: Union[None, bool, Literal['check']] = None,
        force_reload: bool = False,
        verbose: bool = True,
        skip_validation: bool = False,
        device: TorchDevice = None,
        **kwargs: Any) -> torch.nn.Module:

    orig_dir = torch.hub.get_dir()

    try:
        torch.hub.set_dir(model_dir)
        module = torch.hub.load(repo_or_dir, model, *args,
                                source=source,
                                trust_repo=trust_repo,
                                force_reload=force_reload,
                                verbose=verbose,
                                skip_validation=skip_validation,
                                **kwargs)

        if not isinstance(module, torch.nn.Module):
            from util import typename, typename_of
            raise TypeError(
                f'Expected type `{typename(torch.nn.Module)}` for the object obtained from the PyTorch hub. The actual '
                f'object type however was `{typename_of(module)}`.')

        if device is not None:
            module.to(device=device)

        return module
    finally:
        torch.hub.set_dir(orig_dir)


class FeatureBackboneInvoker(ABC):
    @abstractmethod
    def invoker(self) -> Callable[[torch.Tensor], torch.Tensor]:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    @abstractmethod
    def identifier(self) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def feature_count(self) -> int:
        ...

    @property
    @abstractmethod
    def input_count(self) -> int:
        ...

    @abstractmethod
    def feature_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        ...

    @abstractmethod
    def to(self, device: TorchDevice) -> None:
        ...


class FeatureBackboneFactory(ABC):
    @abstractmethod
    def create_invoker(
            self,
            transform: Union[bool, Callable[[Union[torch.Tensor, PIL.Image.Image, np.ndarray]], torch.Tensor]] = True) \
            -> FeatureBackboneInvoker:
        ...
