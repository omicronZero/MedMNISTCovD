from nn_lit import ClassifierModel

import torch

from typing import Optional, Union, Any, Callable, Literal

import lightning.pytorch as pl


class RegularClassifierModel(ClassifierModel):
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
                 trainer_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(result_dir, model_identifier, labels, trainer, loss_function, class_proportions,
                         training_batch_size, val_test_batch_size, trainer_kwargs)

        self._model = model

    def _forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self._model(*input)
