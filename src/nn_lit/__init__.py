import os.path
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union, Mapping, Callable, Literal, NamedTuple, overload, Sequence

from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from typing_extensions import Self

import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Metric

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from torchutil import TorchDevice


@dataclass(frozen=True)
class ModelInitializationParameters:
    result_dir: str
    model_identifier: str
    labels: tuple[str, ...]
    trainer: Optional[pl.Trainer]
    training_batch_size: int
    val_test_batch_size: int
    trainer_kwargs: Optional[dict[str, Any]]


default_optimizer_factory = partial(torch.optim.Adam, lr=1e-3)

_monitored_metric, _monitored_metric_mode = 'val/loss', 'min'


def model_checkpoint(params: ModelInitializationParameters) -> pl.callbacks.ModelCheckpoint:
    i = 1
    while True:
        directory = os.path.join(params.result_dir, 'lit', params.model_identifier, f'model-{i}')

        if not os.path.exists(directory):
            break
        i += 1

    os.makedirs(directory, exist_ok=True)

    return pl.callbacks.ModelCheckpoint(directory,
                                        monitor=_monitored_metric, mode=_monitored_metric_mode,
                                        save_top_k=1)


default_callbacks: list[Callable[[ModelInitializationParameters], pl.Callback]] = [
    lambda _: pl.callbacks.EarlyStopping(_monitored_metric, patience=15, check_on_train_epoch_end=False,
                                         strict=False, mode=_monitored_metric_mode),
    model_checkpoint,
    lambda _: pl.callbacks.LearningRateMonitor(logging_interval='step')
]


class ModelEvaluation(NamedTuple):
    forwarded: torch.Tensor
    prediction: torch.Tensor
    loss: torch.Tensor


def _validate_data(data: Dataset, name: str) -> None:
    len_method = getattr(data, '__len__', None)

    if len_method is not None:
        if len_method() == 0:
            raise ValueError(f'`{name}` must provide at least one datapoint.')


class PredictionTruthPair(NamedTuple):
    prediction: torch.Tensor
    targets: torch.Tensor


def _move_metric_collection(c: MetricCollection, device: TorchDevice) -> None:
    for m in c.values():
        m.to(device=device)


class BinaryBalancedAccuracyMetric(Metric):
    def __init__(self, threshold: float = .5) -> None:
        super().__init__()

        from torchmetrics.classification import MulticlassAccuracy

        self._inner_metric = MulticlassAccuracy(num_classes=2)
        self._threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.dtype.is_floating_point:
            preds = (preds > self._threshold).int()

        self._inner_metric.update(preds, target)

    def compute(self) -> torch.Tensor:
        return self._inner_metric.compute()


_tensorboard_launched: bool = False


class BaseModel(lightning.pytorch.LightningModule, ABC):
    def __init__(self,
                 result_dir: str,
                 model_identifier: str,
                 labels: tuple[str, ...],
                 trainer: Optional[pl.Trainer] = None,
                 training_batch_size: int = 32,
                 val_test_batch_size: int = 256,
                 trainer_kwargs: Optional[dict[str, Any]] = None,
                 auto_log: bool = True) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=type(self).ignored_hyperparameter_args())

        self._training_batch_size = training_batch_size
        self._val_test_batch_size = val_test_batch_size

        self._labels = labels

        try:
            import wandb
            project_name = model_identifier if wandb.run is None else wandb.run.project_name()
        except ImportError:
            project_name = model_identifier

        self._project_name = project_name
        self._model_identifier = model_identifier

        if trainer is None:
            params = dict(type(self)._default_trainer_parameters())

            if trainer_kwargs is not None:
                params.update(trainer_kwargs)

            if 'logger' not in params:
                if not auto_log:
                    params['logger'] = False
                else:
                    uses_wandb = False

                    import wandb

                    if wandb.run is None:
                        print('Not logging to wandb since it is not initialized.')
                    else:
                        uses_wandb = True
                        from lightning.pytorch.loggers import WandbLogger
                        params['logger'] = WandbLogger(
                            project=project_name,
                            dir=os.path.join(result_dir, 'wandb'),
                            job_type='train')

                    if not uses_wandb:
                        supports_tensorboard = True
                        try:
                            import tensorboard
                        except ModuleNotFoundError:
                            supports_tensorboard = False
                            import warnings
                            warnings.warn('`tensorboard` package is not available.')

                        if supports_tensorboard:
                            from lightning.pytorch.loggers import TensorBoardLogger

                            from util import next_dir
                            log_root_dir = os.path.join(result_dir, 'tensorboard')
                            log_dir = next_dir(log_root_dir, lambda run: f'Run {run}')

                            print(f'Tensorboard log-directory:\n{os.path.normpath(log_root_dir)}')

                            logger = TensorBoardLogger(save_dir=log_dir)
                            params['logger'] = logger

                            global _tensorboard_launched
                            if not _tensorboard_launched:
                                try:
                                    import tensorboard.program as tb

                                    # If installed, we launch tensorboard itself and return the url
                                    tensorboard = tb.TensorBoard()
                                    tensorboard.configure((None, '--logdir', log_root_dir))
                                    try:
                                        url = tensorboard.launch()
                                        _tensorboard_launched = True
                                    except tb.TensorBoardServerException as ex:
                                        import warnings
                                        warnings.warn(str(ex))
                                        url = None
                                        pass

                                    print(f'Tensorboard url: {url}')
                                except ModuleNotFoundError as ex:
                                    import warnings
                                    warnings.warn(str(ex))

            callbacks: list[
                Union[pl.Callback, Callable[[ModelInitializationParameters], pl.Callback]]] = default_callbacks

            if 'callbacks' in params:
                callbacks = params.pop('callbacks')

            state = ModelInitializationParameters(result_dir, model_identifier, labels, trainer, training_batch_size,
                                                  val_test_batch_size, trainer_kwargs)
            callbacks = [callback if isinstance(callback, pl.Callback) else callback(state)
                         for callback in callbacks]

            # find next non-existent directory

            from util import next_dir
            root_dir = next_dir(os.path.join(result_dir, 'lit'), lambda i: f'{model_identifier}_{i}')

            trainer = pl.Trainer(**params, callbacks=callbacks, enable_checkpointing=True,
                                 default_root_dir=root_dir)

            self.trainer = trainer

        if len(self.trainer.checkpoint_callbacks) > 1:
            import warnings
            warnings.warn('Found more than one checkpoint callback.')

        self.__metrics: dict[str, MetricCollection] = {}
        self.__last_metrics: dict[str, Optional[dict[str, float]]] = {}

    def _dataloader(self, data: Dataset, is_training: bool) -> DataLoader:
        return DataLoader(data,
                          batch_size=self._training_batch_size if is_training else self._val_test_batch_size,
                          shuffle=is_training,
                          generator=torch.Generator(device=self.device) if is_training else None)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def root_dir(self) -> str:
        return self.trainer.default_root_dir

    @classmethod
    def ignored_hyperparameter_args(cls) -> list[str]:
        return []

    @classmethod
    def _default_trainer_parameters(cls) -> dict[str, Any]:
        return dict(
            max_epochs=250,
            precision='32',  # Alternative would be '16-mixed'
            #     strategy='deepspeed_stage_2',
            log_every_n_steps=30
        )

    @property
    def model_identifier(self) -> str:
        return self._model_identifier

    def _get_metrics(self, mode: Literal['train', 'val', 'test']) -> MetricCollection:
        metrics = self.__metrics.get(mode)

        if metrics is None:
            self.__metrics[mode] = metrics = self._create_metrics(mode).to(device=self.device)

        return metrics

    def _reset_metrics(self, mode: Literal['train', 'val', 'test']):
        self.__metrics.pop(mode)

    def _create_metrics(self, subset: Union[Literal['train', 'val', 'test'], str]) -> MetricCollection:
        metrics = MetricCollection(())

        # metrics[subset + '/Loss'] = torchmetrics.MeanMetric()

        return metrics

    def to(self, *args: Any, **kwargs: Any) -> Self:
        s = super().to(*args, **kwargs)

        device = self.device

        for metric in self.__metrics.values():
            metric.to(device=device)

        return s

    @abstractmethod
    def criterion(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def forward(self, input: tuple[torch.Tensor, ...]) -> torch.Tensor:
        ...

    def forward(self, *input: Union[torch.Tensor, tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if len(input) == 1 and not isinstance(input[0], torch.Tensor):
            input = input[0]

        if not all(isinstance(inp, torch.Tensor) for inp in input):
            raise ValueError('All values in `input` must be tensors.')

        return self._forward(*input)

    @abstractmethod
    def _forward(self, *input: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def forwarded_to_prediction(self, forwarded: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, *input: torch.Tensor) -> torch.Tensor:
        return self.forwarded_to_prediction(self.forward(*input))

    def predict_test(self, dataset: Dataset) -> PredictionTruthPair:
        _validate_data(dataset, 'dataset')

        self.test(dataset)

        test_loader = self._dataloader(dataset, False)

        labels = []
        preds = []

        for inp, lbl in test_loader:
            if isinstance(inp, torch.Tensor):
                inp = (inp,)

            pred = self.predict(*(x.to(device=self.device) for x in inp))
            if len(pred) != len(lbl):
                raise RuntimeError('Prediction and label must have the same length.')

            labels.append(lbl.to(device=self.device))
            preds.append(pred.detach())

        return PredictionTruthPair(torch.cat(preds), torch.cat(labels))

    @overload
    def fit(self, training_data: Dataset, validation_data: Dataset, test_data: Dataset) -> Mapping[str, float]:
        ...

    @overload
    def fit(self, training_data: Dataset, validation_data: Dataset, test_data: Literal[None] = None) -> None:
        ...

    @overload
    def fit(self, training_data: Dataset, validation_data: Dataset, test_data: Optional[Dataset] = None) \
            -> Optional[Mapping[str, float]]:
        ...

    def fit(self, training_data: Dataset, validation_data: Dataset, test_data: Optional[Dataset] = None) \
            -> Optional[Mapping[str, float]]:
        self.__init_metrics()

        _validate_data(training_data, 'training_data')
        _validate_data(validation_data, 'validation_data')
        _validate_data(test_data, 'test_data')

        training_loader = self._dataloader(training_data, True)
        validation_loader = self._dataloader(validation_data, False)

        device = self.device

        self.trainer.fit(self, training_loader, validation_loader)

        checkpoint_cb = self.trainer.checkpoint_callback

        if checkpoint_cb is not None and isinstance(checkpoint_cb, ModelCheckpoint):
            new_inst = type(self).load_from_checkpoint(checkpoint_cb.best_model_path, map_location=device)

            self.load_state_dict(new_inst.state_dict())

        if self.device != device:
            self.to(device=device)

        if test_data is None:
            return None

        return self.test(test_data)

    def __init_metrics(self) -> None:
        self._get_metrics('train')
        self._get_metrics('val')
        self._get_metrics('test')

    def test(self, dataset: Dataset) -> Mapping[str, float]:
        _validate_data(dataset, 'dataset')

        test_loader = self._dataloader(dataset, False)

        device = self.device

        result = self.trainer.test(self, test_loader)[0]

        if self.device != device:
            self.to(device=device)

        return result

    def _create_optimizer(self) -> torch.optim.Optimizer:
        return default_optimizer_factory(self.parameters())

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self._create_optimizer()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        #                                                                  100,
        #                                                                  eta_min=1e-5)
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        if scheduler is None:
            return [optimizer]
        else:
            return [optimizer], [scheduler]

    def transform(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return images, targets

    # def to(self, device: TorchDevice, *args: Any, **kwargs: Any) -> Self:
    #    self._get_metrics('train').to(device)
    #    self._get_metrics('val').to(device)
    #    self._get_metrics('test').to(device)
    #    return super().to(device, *args, **kwargs)

    def _on_predicted(self,
                      mode: Literal['train', 'val', 'test'],
                      inputs: torch.Tensor,
                      targets: torch.Tensor,
                      prediction: torch.Tensor) -> None:
        pass

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets

    def _invoke_on(self,
                   mode: Literal['train', 'val', 'test'],
                   inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                   targets: torch.Tensor,
                   on_step: Optional[bool] = None,
                   on_epoch: Optional[bool] = None) -> ModelEvaluation:
        no_grad = None

        if mode in ('val', 'test'):
            assert not self.training
            no_grad = torch.no_grad()
            no_grad.__enter__()
        else:
            assert self.training

        try:
            y_forwarded = self.forward(inputs)
            y_pred = self.forwarded_to_prediction(y_forwarded)

            loss = self.criterion(y_forwarded, targets)

            self.log_dict({f'{mode}/loss': loss}, on_step=on_step, on_epoch=on_epoch)

            metrics = self._get_metrics(mode)

            y_pred, targets = self._prepare_output(y_pred, targets)

            y_pred_detached = y_pred.detach()

            metrics.update(y_pred_detached, targets)

            self._on_predicted(mode, inputs, targets, y_pred_detached)
        except BaseException as ex:
            if mode in ('val', 'test'):
                no_grad.__exit__(type(ex), ex, ex.__traceback__)

            raise

        if mode in ('val', 'test'):
            no_grad.__exit__(None, None, None)

        # self._report(mode, on_step=on_step, on_epoch=on_epoch)

        return ModelEvaluation(y_forwarded, y_pred, loss)

    def _prepare_output(self, y_pred: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return y_pred, targets

    def training_step(self,
                      batch: tuple[Union[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        inputs, targets = batch

        return self._invoke_on('train', inputs, targets).loss

    def validation_step(self,
                        batch: tuple[Union[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        inputs, targets = batch

        return self._invoke_on('val', inputs, targets).loss

    def test_step(self,
                  batch: tuple[Union[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        inputs, targets = batch

        return self._invoke_on('test', inputs, targets).loss

    def _update_last_metrics(self, mode: Literal['train', 'val', 'test']) -> None:
        m = self.__metrics.get(mode)

        if m is None:
            if mode in self.__last_metrics:
                self.__last_metrics.pop(mode)
        else:
            self.__last_metrics[mode] = {k: v.item() if isinstance(v, torch.Tensor) else float(v)
                                         for k, v in m.compute().items()}

    def last_metrics(self, mode: Literal['train', 'val', 'test'], strip_mode: bool = True) -> dict[str, float]:
        src = self.__last_metrics.get(mode)

        if src is None:
            return {}

        return {k[len(mode) + 1:] if strip_mode else k: v for k, v in src.items()}

    def on_train_epoch_end(self) -> None:
        self._report('train')
        self._update_last_metrics('train')
        self._reset_metrics('train')

    def on_validation_epoch_end(self) -> None:
        self._report('val')
        self._update_last_metrics('val')
        self._reset_metrics('val')

    def on_test_epoch_end(self) -> None:
        self._report('test')
        self._update_last_metrics('test')
        self._reset_metrics('test')

    def _report(self,
                mode: Literal['train', 'val', 'test'],
                on_step: Optional[bool] = None,
                on_epoch: Optional[bool] = None) -> None:
        metrics = self._get_metrics(mode)
        values = metrics.compute()

        values['epoch'] = self.current_epoch

        self.log_dict(values, on_step=on_step, on_epoch=on_epoch)

        # metrics.reset()

        # for name, metric in metrics.items():
        #     try:
        #         self.log(name, metric.compute().item(), on_step=on_step, on_epoch=on_epoch)
        #     except ValueError as ex:
        #         if 'No samples to concatenate' in ex.args:
        #             self.log(name, float('nan'), on_step=on_step, on_epoch=on_epoch)
        #         else:
        #             raise

    def on_after_batch_transfer(self, batch: tuple[torch.Tensor, torch.Tensor], dataloader_idx: int) -> Any:
        images, targets = batch
        images, targets = self.transform(images, targets)

        return images, targets


class ClassifierModel(BaseModel, ABC):
    def __init__(self,
                 result_dir: str,
                 model_identifier: str,
                 labels: tuple[str, ...],
                 trainer: Optional[pl.Trainer] = None,
                 loss_function: Union[
                     torch.nn.Module,
                     Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     Literal[
                         'bce', 'bce_logits',
                         'ce', 'ce_logits',
                         'auto_ce', 'auto_ce_logits']
                 ] = 'auto_ce_logits',
                 class_proportions: Optional[torch.Tensor] = None,
                 training_batch_size: int = 32,
                 val_test_batch_size: int = 256,
                 trainer_kwargs: Optional[dict[str, Any]] = None,
                 auto_log: bool = True) -> None:
        super().__init__(result_dir, model_identifier, labels, trainer, training_batch_size, val_test_batch_size,
                         trainer_kwargs, auto_log=auto_log)

        classification_mode = None

        if not callable(loss_function):
            from layers.losses import get_classifier_loss

            classification_mode, loss_function = get_classifier_loss(loss_function, len(labels),
                                                                     class_proportions=class_proportions)

        self._classification_mode = classification_mode
        self._loss_function = loss_function

    def _create_metrics(self, subset: Union[Literal['train', 'val', 'test'], str]) -> MetricCollection:
        import torchmetrics
        metrics = super()._create_metrics(subset)

        task: Literal['binary', 'multiclass'] = 'binary' if self.classification_mode == 'binary' else 'multiclass'
        average: Literal['macro', 'micro'] = 'micro' if self.classification_mode == 'binary' else 'macro'

        num_classes = len(self._labels)

        if self.classification_mode == 'binary':
            bacc = BinaryBalancedAccuracyMetric()
        else:
            bacc = torchmetrics.Accuracy('multiclass',
                                         num_classes=num_classes,
                                         average='macro')

        metric_dict = {
            subset + '/Accuracy': torchmetrics.Accuracy(task, num_classes=num_classes),
            subset + '/Balanced accuracy': bacc,
            subset + '/AUROC': torchmetrics.AUROC(task, num_classes=num_classes),
            subset + '/Recall': torchmetrics.Recall(task, num_classes=num_classes, average=average),
            subset + '/Precision': torchmetrics.Precision(task, num_classes=num_classes, average=average),
            subset + '/F1': torchmetrics.F1Score(task, num_classes=num_classes, average=average),
        }

        for name, metric in metric_dict.items():
            metrics[name] = metric

        return metrics

    def _prepare_output(self, y_pred: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.classification_mode == 'binary':
            from util.shapes import shape_to_string
            if y_pred.ndim not in (1, 2) or (y_pred.ndim == 2 and y_pred.shape[-1] != 1):
                raise ValueError('Expected model output for binary classification to be of shape (N,) or (N, 1). Got '
                                 f'shape {shape_to_string(y_pred.shape)}.')

            if y_pred.ndim == 2:
                y_pred = y_pred.flatten()
        else:
            if y_pred.ndim != 2:
                from util.shapes import shape_to_string
                raise ValueError('Expected model output for multi-class classification to be of shape (N, C). Got '
                                 f'shape {shape_to_string(y_pred.shape)}.')

        if targets.ndim not in (1, 2) or (targets.ndim == 2 and targets.shape[-1] != 1):
            from util.shapes import shape_to_string
            raise ValueError('Expected targets for single-target classification to be of shape (N,) or (N, 1). Got '
                             f'shape {shape_to_string(targets.shape)}.')

        if targets.ndim == 2:
            targets = targets.flatten()

        return y_pred, targets

    @property
    def classification_mode(self) -> Optional[Literal['binary', 'categorical']]:
        return self._classification_mode

    def criterion(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Shapes:
        # bce/bin. focal: input and target must have the same shape
        # ce/cat. focal:
        #   - option 1 (1-dim. loss): input: N, C; target: (N,)
        #   - option 2 (K-dim. loss): input: (N, C, d[1], ..., d[K]); target: (N, d[1], ..., d[K])
        # --> bin. loss has one dim. less than cat. loss

        if self.classification_mode == 'binary':
            # be gracious if the binary loss shape is just different by a single dim (e.g., (N, 1) and (N,))
            # --> categorical loss would expect (N, 1), we can merge the two
            if y_true.ndim != y_pred_logits.ndim:
                if y_pred_logits.ndim == y_true.ndim + 1 and y_pred_logits.shape[-1] == 1:
                    y_pred_logits = y_pred_logits.squeeze(-1)
                elif y_true.ndim == y_pred_logits.ndim + 1 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)

        return self._loss_function(y_pred_logits, y_true)

    def forwarded_to_prediction(self, forwarded: torch.Tensor) -> torch.Tensor:
        if self.classification_mode == 'binary':
            return forwarded.sigmoid()
        else:
            return forwarded.softmax(dim=-1)
