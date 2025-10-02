import json
from typing import Literal, NamedTuple, Optional
import os

import numpy as np

import lightning.pytorch as pl

import medmnist_dataset
from common import handcrafted_features, cov_descr
from nn_lit import PredictionTruthPair, model_checkpoint
from nn_lit.spdnets import BasicSPDNet, SPDNetClassifierModel
from torchutil import TorchDevice

import torch
from torch.utils.data import Dataset

default_window_size = 21
default_stride = 12


class FitSPDNetResult(NamedTuple):
    eval_metrics: dict[str, float]
    test_results: PredictionTruthPair


def fit_spdnet(
        model_directory: str,
        logging_directory: str,
        project_name: str,
        dataset: medmnist_dataset.MedMNIST2D,
        features: str,
        training_data: Dataset,
        validation_data: Dataset,
        test_data: Dataset,
        class_proportions: Optional[torch.Tensor],
        feature_count: int,
        class_count: int,
        device: TorchDevice = None,
        train_batch_size: int = 32,
        val_batch_size: int = 256,
        max_epochs: int = ...,
        repeats: int = ...,
        early_stopping_patience: int = ...,
        early_stopping_epsilon: float = 0.,
        learning_rate: float = 1e-3,
        best_pick_criterion: Literal[
            'Accuracy',
            'Balanced accuracy',
            'AUROC',
            'Recall',
            'Precision',
            'F1'
        ] = 'Balanced accuracy') -> tuple[dict[str, float], PredictionTruthPair]:
    if dataset in (medmnist_dataset.MedMNIST2D.path_mnist,
                   medmnist_dataset.MedMNIST2D.organ_a_mnist,
                   medmnist_dataset.MedMNIST2D.oct_mnist,
                   medmnist_dataset.MedMNIST2D.tissue_mnist):
        if repeats is ...:
            repeats = 1
        if max_epochs is ...:
            max_epochs = 20
        if early_stopping_patience is ...:
            early_stopping_patience = 15
    else:
        if repeats is ...:
            repeats = 5
        if max_epochs is ...:
            max_epochs = 250
        if early_stopping_patience is ...:
            early_stopping_patience = 15

    assert repeats > 0
    assert max_epochs > 0
    assert early_stopping_patience > 0

    target_criterion = 'val/' + best_pick_criterion

    best_test = None
    best_prediction = None

    try:
        import wandb
    except ModuleNotFoundError:
        wandb = None

    for i in range(1, repeats + 1):
        result_dir = os.path.abspath(os.path.join(model_directory, dataset.dataset_name, features, str(i)))

        print(f'Attempt {i}: {result_dir}')

        metrics_file = os.path.join(result_dir, 'metrics.json')
        prediction_file = os.path.join(result_dir, 'prediction.npy')
        target_file = os.path.join(result_dir, 'target.npy')

        metrics = None

        try:
            # attempt to load the metrics from the json file. It only exists after successful training
            with open(metrics_file, 'r', encoding='utf-8') as stream:
                metrics = json.load(stream)

            prediction = PredictionTruthPair(torch.tensor(np.load(prediction_file)),
                                             torch.tensor(np.load(target_file)))
        except FileNotFoundError:
            default_model_checkpoint_factory = model_checkpoint
            spdnet_type = SPDNetClassifierModel
            optimizer_kwargs = {
                'lr': learning_rate,
                'optimizer': torch.optim.Adam
            }
            backbone = BasicSPDNet(feature_count,
                                   1 if class_count == 2 else class_count,
                                   device=device)

            if wandb is not None:
                wandb_dir = os.path.join(logging_directory, 'wandb', dataset.dataset_name, features, str(i))
                os.makedirs(wandb_dir, exist_ok=True)
                run = wandb.init(dir=wandb_dir,
                                 project=project_name,
                                 name='-'.join([project_name, dataset.dataset_name, features, f'run {i}']),
                                 group='-'.join([project_name, dataset.dataset_name, features]))
                run.__enter__()
            else:
                run = None

            # Loading failed. We train a new model and retrieve its metrics instead
            spdnet = spdnet_type(result_dir,
                                 f'spdnet-{dataset.dataset_name}-{features}',
                                 backbone,
                                 dataset.labels,
                                 loss_function='auto_ce_logits',
                                 class_proportions=class_proportions,
                                 training_batch_size=train_batch_size,
                                 val_test_batch_size=val_batch_size,
                                 trainer_kwargs={
                                     'max_epochs': max_epochs,
                                     'callbacks': [
                                         lambda _: pl.callbacks.EarlyStopping(
                                             target_criterion,
                                             patience=early_stopping_patience,
                                             min_delta=early_stopping_epsilon,
                                             check_on_train_epoch_end=False,
                                             strict=False,
                                             mode='min'),
                                         default_model_checkpoint_factory
                                     ]
                                 },
                                 optimizer_kwargs=optimizer_kwargs)

            device = torch.empty(()).device
            spdnet.to(device=device)

            try:
                # fit on train/val data and predict on test set
                metrics = dict(spdnet.fit(training_data, validation_data, test_data))

                # we keep the validation metrics from the last epoch (we'll use them below)
                val_metrics = spdnet.last_metrics('val', strip_mode=False)

                assert metrics.keys().isdisjoint(val_metrics.keys())

                prediction = spdnet.predict_test(test_data)

                if run is not None:
                    run.__exit__(None, None, None)
            except BaseException as ex:
                if run is not None:
                    run.__exit__(type(ex), ex, ex.__traceback__)
                raise

            metrics.update(val_metrics)

            # We've finished computing our metrics, save them so that we can reuse them the next time.
            with open(metrics_file, 'w', encoding='utf-8') as stream:
                json.dump(metrics, stream)

            np.save(prediction_file, prediction.prediction.numpy(force=True))
            np.save(target_file, prediction.targets.numpy(force=True))

        # we pick the best-performing model regarding the criterion the user indicated. Note that we ensured above that
        # the validation set is used, not the test set!
        if (best_test is None or
                metrics[target_criterion] > best_test[target_criterion]):
            best_test = metrics
            best_prediction = prediction

    assert best_test is not None
    assert best_prediction is not None

    return best_test, best_prediction


def spdnet_hc_cov(images: torch.Tensor, window_size: int = ..., stride: int = ...) -> torch.Tensor:
    if window_size is ...:
        window_size = default_window_size
    if stride is ...:
        stride = default_stride

    images = torch.tensor(handcrafted_features(images.numpy(force=True)), dtype=images.dtype, device=images.device)

    from torchutil.views import sliding_window
    # images: (*N)xCxHxW
    # windows: (*N)xCxhxwxH'xW'
    windows = sliding_window(images, (window_size, stride), (window_size, stride), axes=(-2, -1))

    # --> (*N)x(C*h*w)xH'xW'
    windows = windows.flatten(-5, -3)

    # QxQ with Q = C * h * w
    cov = cov_descr(windows)

    return cov
