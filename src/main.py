import os
from typing import Iterable, Callable, TypeVar, Sequence, Any, cast

import config

if config.load_user_config() is None:
    import setup

    setup.setup()

user_config = config.load_user_config()

if user_config is None:
    from util.setuputil import press_enter_to_exit

    print('Something went wrong. Restart the application or file a bug report if you think there is an issue.')
    press_enter_to_exit()

is_first = True

while True:
    try:
        import numpy as np
        import torch
        import torch.utils.data
        from torch.utils.data import Dataset
        import medmnist as medmnist_data
        import segment_anything
        break
    except (ModuleNotFoundError, ImportError) as ex:
        if not is_first:
            print('The setup failed. This may either be due to an error or due to an error in one of the dependencies:'
                  '\n' +
                  str(ex))
            from util.setuputil import press_enter_to_exit
            press_enter_to_exit()

        is_first = False

        import setup
        setup.setup()


import datetime

import medmnist_dataset

from data import default_transform
from experiments import reported_metric_scores, to_dataframe, reported_metrics, prepare_datasets, \
    training_set_label_proportions
from common import cov_descr
from experiments.spdnet import spdnet_hc_cov, fit_spdnet


def medmnist_dir() -> str:
    return os.path.join(user_config.dataset_dir, 'medmnist')


run_date = datetime.datetime.now()


# do some required project-wise general initializations
if torch.cuda.is_available():
    torch.set_default_device('cuda')

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'


_T = TypeVar('_T')


def map_dataset_to_handcrafted_covs(data: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    from common import to_feature_covs

    if isinstance(data, medmnist_data.dataset.MedMNIST2D):
        images = data.imgs
        labels = data.labels
    else:
        images, labels = [], []

        for img, lbl in data:
            images.append(img)
            labels.append(lbl)

        images = np.stack(images)
        labels = np.stack(labels)

    return to_feature_covs(images), torch.tensor(labels)


_remark_file = None


def remark(text: str) -> None:
    global _remark_file

    if _remark_file is None:
        _remark_file = open(os.path.join(user_config.result_dir, 'REMARK.txt'), 'a')

        _remark_file.write('\n')
        _remark_file.write(f'[{run_date.strftime("%Y-%m-%d %H-%M-%S")}]:\n')

    text = text + '\n'
    _remark_file.write(text)

    import sys
    sys.stderr.write(text)


def spdnet_hc_feature_count(channels: int, window_size: int = ..., stride: int = ...) -> int:
    from experiments import image_size
    # we can't use a meta-tensor here since the features are computed on numpy
    sample = torch.rand(channels, image_size, image_size, dtype=torch.float32)

    cov = spdnet_hc_cov(sample, window_size, stride)

    return cov.shape[-1]


reports = {}


def add_report(dataset: medmnist_dataset.MedMNIST2D,
               technique: str,
               features: str,
               prediction: torch.Tensor,
               ground_truth: torch.Tensor,
               decision_threshold: float = .5,
               debug: bool = False) -> None:
    pred_np = prediction.numpy(force=True)
    gt_np = ground_truth.numpy(force=True)

    key = (technique, features)

    technique_reports = reports.get(key)

    if technique_reports is None:
        reports[key] = technique_reports = []

    if np.issubdtype(pred_np.dtype, np.floating):
        scores_np = pred_np
        if pred_np.shape[-1] == 1:
            pred_np = cast(np.ndarray, pred_np > decision_threshold).astype(np.int64)
        elif prediction.ndim > 1:
            pred_np = np.argmax(pred_np, axis=-1)
    else:
        scores_np = None

    values = [metric(gt_np, pred_np) for _, metric in reported_metrics]

    if scores_np is None:
        values.extend([None] * len(reported_metric_scores))
    else:
        values.extend([metric(gt_np, scores_np) for _, metric in reported_metric_scores])

    technique_reports.append(values)

    result_dir = user_config.result_dir

    if debug:
        result_dir = os.path.join(result_dir, 'debug')

    os.makedirs(result_dir, exist_ok=True)

    for (technique, features), results in reports.items():
        try:
            to_dataframe(results).to_csv(os.path.join(result_dir, technique + ' ' + features + '.csv'), sep=';')
        except:
            pass


def handle_traditional_handcrafted(ds: medmnist_dataset.MedMNIST2D, debug: bool = False) -> None:
    print(f'Traditional, handcrafted on dataset: {ds.dataset_name}')

    from experiments.traditional import mdrm, tslda

    train_ds, test_ds = prepare_datasets(medmnist_dir(), ds, False,
                                         debug=debug)

    with torch.no_grad():
        train_data = map_dataset_to_handcrafted_covs(train_ds)
        test_data = map_dataset_to_handcrafted_covs(test_ds)

        train_covs, train_lbl = train_data
        test_covs, test_lbl = test_data

        assert train_lbl.shape[1:] == test_lbl.shape[1:]
        assert train_lbl.ndim == 2
        assert train_lbl.shape[-1] == 1

        train_lbl = train_lbl.squeeze(-1)
        test_lbl = test_lbl.squeeze(-1)

        assert len(train_covs) == len(train_lbl)
        assert len(test_covs) == len(test_lbl)

    # MDRM
    prediction = mdrm(train_covs, train_lbl, test_covs, remark)
    add_report(ds, 'MDRM', 'Handcrafted', prediction, test_lbl, debug=debug)

    # TSLDA
    prediction = tslda(train_covs, train_lbl, test_covs, remark, basepoint='mean')
    add_report(ds, 'TSLDA at mean', 'Handcrafted', prediction, test_lbl, debug=debug)


def handle_traditional_gve(ds: medmnist_dataset.MedMNIST2D,
                           subsamples: int = 256,
                           debug: bool = False) -> None:
    print(f'Traditional, general vision encoder features on dataset: {ds.dataset_name}')

    from common.gve import pretrain_pca, PretrainedFeatures, image_to_pca, image_from_pca
    from common import take_samples, pil_to_numpy, tqdm, cov_descr
    from experiments.traditional import mdrm, tslda

    train_ds, test_ds = prepare_datasets(medmnist_dir(), ds, False,
                                         debug=debug)

    for gve in ('dino', 'medsam'):
        print(f'  {gve}.')
        print('    Pretraining PCA...')

        model = PretrainedFeatures(user_config.model_dir, gve)

        sample_images = [pil_to_numpy(img) for img, _ in take_samples(cast(Sequence, train_ds), subsamples)]

        pca = pretrain_pca(sample_images, model)
        del sample_images

        train_covs = []
        train_lbl = []

        test_covs = []
        test_lbl = []

        for mode, src, tgt_covs, tgt_lbl in (('training', train_ds, train_covs, train_lbl),
                                             ('test', test_ds, test_covs, test_lbl)):
            print(f'    Computing {mode} covariance descriptors...')

            for img, lbl in tqdm(cast(Iterable, src)):
                # to numpy
                img = pil_to_numpy(img)

                # compute feature image
                fimg = model(torch.tensor(img))

                C, H, W = fimg.shape

                # apply PCA to feature image
                fimg = torch.tensor(image_from_pca(pca.transform(image_to_pca(fimg)), H, W))

                # compute covariance descriptor
                cov = cov_descr(fimg)

                # add to respective field
                tgt_covs.append(cov)

                tgt_lbl.append(torch.tensor(lbl))

        train_covs = torch.stack(train_covs)
        train_lbl = torch.stack(train_lbl)
        test_covs = torch.stack(test_covs)
        test_lbl = torch.stack(test_lbl)

        # MDRM
        prediction = mdrm(train_covs, train_lbl, test_covs, remark)
        add_report(ds, 'MDRM', gve, prediction, test_lbl, debug=debug)

        # TSLDA
        prediction = tslda(train_covs, train_lbl, test_covs, remark, basepoint='mean')
        add_report(ds, 'TSLDA at mean', gve, prediction, test_lbl, debug=debug)


def handle_spdnet_handcrafted(ds: medmnist_dataset.MedMNIST2D,
                              cache: bool = True,
                              debug: bool = False) -> None:
    print(f'SPDNet, handcrafted on dataset: {ds.dataset_name}')
    cache_dir = get_cache_dir(ds, 'handcrafted') if cache else None

    print('  Preparing dataset...')

    mapper = data_mapper(spdnet_hc_cov, flatten_label=True)
    post = None
    column_names = ('images', 'labels')

    training_set, validation_set, test_set = prepare_datasets(
        medmnist_dir(),
        ds,
        require_validation_sets=True,
        cache_dir=cache_dir,
        transform=mapper,
        uncached_transform=post,
        column_names=column_names,
        debug=debug)

    print('  Fitting and evaluating...')
    metrics, predictions = fit_spdnet(user_config.model_dir,
                                      ds, 'handcrafted', training_set, validation_set, test_set,
                                      class_proportions=training_set_label_proportions(medmnist_dir(), ds),
                                      feature_count=spdnet_hc_feature_count(ds.n_channels),
                                      class_count=len(ds.labels),
                                      max_epochs=1 if debug else ...)

    add_report(ds, 'SPDNet', 'Handcrafted', predictions.prediction, predictions.targets, debug=debug)


def handle_spdnet_gve(ds: medmnist_dataset.MedMNIST2D,
                      cache: bool = True,
                      debug: bool = False) -> None:
    from common.gve import PretrainedFeatures
    print(f'SPDNet, general vision encoder features on dataset: {ds.dataset_name}')
    for gve in ('dino', 'medsam'):
        cache_dir = get_cache_dir(ds, gve) if cache else None

        print(f'  Fetching/Loading GVE model ({gve})...')
        features = PretrainedFeatures(user_config.model_dir, gve)

        print('  Preparing dataset...')
        mapper = data_mapper(lambda x: cov_descr(features(x)),
                             flatten_label=True)
        post = None
        column_names = ('images', 'labels')

        training_set, validation_set, test_set = prepare_datasets(
            medmnist_dir(),
            ds,
            require_validation_sets=True,
            cache_dir=cache_dir,
            transform=mapper,
            uncached_transform=post,
            column_names=column_names,
            debug=debug)

        print('  Fitting and evaluating...')
        metrics, predictions = fit_spdnet(user_config.model_dir,
                                          ds, gve, training_set, validation_set, test_set,
                                          class_proportions=training_set_label_proportions(medmnist_dir(), ds),
                                          feature_count=features.feature_count,
                                          class_count=len(ds.labels),
                                          max_epochs=1 if debug else ...)

        add_report(ds, 'SPDNet', gve, predictions.prediction, predictions.targets, debug=debug)


def data_mapper(image_transform: Callable[[torch.Tensor], torch.Tensor], flatten_label: bool = False) \
        -> Callable[[Any], tuple[torch.Tensor, torch.Tensor]]:
    def handle(datapoint: Any) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = default_transform(datapoint)

        if flatten_label:
            label = label.squeeze(-1)

        return image_transform(image), label

    return handle


def get_cache_dir(dataset: medmnist_dataset.MedMNIST2D, features: str) -> str:
    return os.path.join(user_config.cache_dir, dataset.dataset_name, features)


def main(print_results: bool = True, debug: bool = False) -> None:
    from experiments import datasets
    run_datasets = datasets

    if debug:
        import sys
        sys.stderr.write('WARNING: Debug mode.')

    for ds in run_datasets:
        handle_traditional_handcrafted(ds, debug=debug)
        handle_traditional_gve(ds, debug=debug)

        handle_spdnet_handcrafted(ds, debug=debug)
        handle_spdnet_gve(ds, debug=debug)

    if print_results:
        for (technique, features), results in reports.items():
            print(technique, str(features).capitalize())
            print(to_dataframe(results).to_string())
            print()


if __name__ == '__main__':
    main()
