from enum import Enum
from types import MappingProxyType
from typing import Type, Union, Any, Mapping, Optional, Protocol, runtime_checkable, Sequence, NamedTuple

import PIL.Image
import medmnist
import numpy as np
import torch.utils.data


class MLTask(Enum):
    """
    Enumerates various machine learning tasks that models may be able to solve.
    """
    classification_binary = 'classification_binary'
    """Classification with two classes."""

    classification_multiary = 'classification_multiary'
    """Classification with an arbitrary number of non-overlapping classes."""

    classification_multi_label_binary = 'classification_multi_label_binary'
    """Classification with an arbitrary number of overlapping possibly classes."""

    ordinal_regression = 'ordinal_regression'
    """Regression with ordinal numbers."""

    @property
    def is_classification(self) -> bool:
        """
        Determines whether the current instance is any type of classification task.

        :return: A Boolean value indicating whether the current instance represents a classification task.
        """
        return self in (MLTask.classification_binary,
                        MLTask.classification_multiary,
                        MLTask.classification_multi_label_binary)

    @property
    def is_regression(self) -> bool:
        """
        Determines whether the current instance is any type of regression task.

        :return: A Boolean value indicating whether the current instance represents a regression task.
        """
        return self in (MLTask.ordinal_regression,)

    @property
    def is_multi_target_classification(self) -> bool:
        return self == MLTask.classification_multi_label_binary

    @property
    def is_multi_target(self) -> bool:
        return self in (MLTask.classification_multi_label_binary,)


class MnistSubset:
    def __init__(self, mnist_type: Type[medmnist.dataset.MedMNIST], **parameters: Any) -> None:
        self._mnist_type = mnist_type
        self._parameters = parameters

    @property
    def mnist_type(self) -> type:
        return self._mnist_type

    @property
    def size(self) -> Optional[int]:
        return self.parameters.get('size')

    @property
    def parameters(self) -> Mapping[str, Any]:
        return MappingProxyType(self._parameters)


class DatasetSplit(Enum):
    train = 'train'
    val = 'val'
    test = 'test'


_task_map = {'binary-class': MLTask.classification_binary,
             'multi-class': MLTask.classification_multiary,
             'multi-label, binary-class': MLTask.classification_multi_label_binary,
             'ordinal-regression': MLTask.ordinal_regression}


@runtime_checkable
class MedMNISTDataSource(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def task(self) -> MLTask: ...

    @property
    def train_val_test_sizes(self) -> tuple[int, int, int]: ...

    @property
    def dataset_sizes(self) -> Mapping[str, int]: ...

    @property
    def datasets(self) -> tuple[str, ...]: ...

    @property
    def n_channels(self) -> int: ...

    @property
    def license(self) -> str: ...

    @property
    def dataset_name(self) -> str: ...

    def info(self) -> MappingProxyType[str, Any]: ...

    def labels(self) -> Optional[tuple[str, ...]]: ...

    @property
    def n_dimensions(self) -> int: ...

    @property
    def is_pil_image(self) -> bool: ...

    @property
    def dataset_class(self) -> type: ...

    def get_subset(self, size: Union[int, tuple[int, ...]]) -> MnistSubset: ...


class MedMNISTShape(NamedTuple):
    images: tuple[int, ...]
    targets: Union[tuple[int], tuple[int, int]]


class _MedMNISTEnumMixin(Enum):

    def info(self) -> MappingProxyType[str, Any]:
        return medmnist.dataset.INFO[self.dataset_name]

    @property
    def labels(self) -> Optional[tuple[str, ...]]:
        label: Optional[dict[str, str]] = self.info().get('label', None)

        if label is None:
            return None

        # we just assume that the documentation has pre-sorted consecutive labels (as it has these)
        return tuple(label.values())

    @property
    def n_channels(self) -> int:
        return self.info()['n_channels']

    @property
    def license(self) -> str:
        return self.info()['license']

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(self.info()['n_samples'])

    @property
    def dataset_sizes(self) -> Mapping[str, int]:
        return MappingProxyType(self.info()['n_samples'])

    def dataset_size_of(self, subset: str) -> int:
        return self.info()['n_samples'][subset]

    @property
    def train_val_test_sizes(self) -> tuple[int, int, int]:
        sizes = self.dataset_sizes

        return sizes['train'], sizes['val'], sizes['test']

    @property
    def task(self) -> MLTask:
        return _task_map[self.info()['task']]

    @property
    def is_task_classification(self) -> bool:
        task = self.task

        return task in (MLTask.classification_binary,
                        MLTask.classification_multi_label_binary,
                        MLTask.classification_multiary)

    @property
    def is_task_regression(self) -> bool:
        return self.task == MLTask.ordinal_regression

    @property
    def is_multi_label(self) -> bool:
        return self.task == MLTask.classification_multi_label_binary

    def get_subset(self, size: Union[int, tuple[int, ...]]) -> MnistSubset:
        unavailable = False
        sz = size
        if isinstance(size, Sequence):
            if len(size) != self.n_dimensions:
                raise ValueError(
                    f'Size must either be a singleton or have length {self.n_dimensions} for this instance.')

            sz = size[0]

            for v in size[1:]:
                if v != sz:
                    unavailable = True
                    break

        if size not in self.available_image_resolutions():
            unavailable = True

        if unavailable:
            raise ValueError(f'Unsupported size {size}.')

        return MnistSubset(self.value, size=sz)

    def data_shape(self, image_resolution: int, channel_last: bool = False, check_available: bool = False) \
            -> MedMNISTShape:
        if check_available and image_resolution not in self.available_image_resolutions():
            raise ValueError('The indicated image size is unavailable.')

        images_shape = (image_resolution,) * self.n_dimensions
        if channel_last:
            images_shape = *images_shape, self.n_channels
        else:
            images_shape = self.n_channels, *images_shape

        if self.task == MLTask.classification_multi_label_binary:
            labels_shape = (len(self.labels),)
        else:
            labels_shape = (1,)

        return MedMNISTShape(images_shape, labels_shape)

    @property
    def dtypes(self) -> tuple[np.dtype, np.dtype]:
        return np.float64, np.int32

    # will be overridden in subclass
    @property
    def dataset_name(self) -> str:
        raise NotImplementedError()

    @property
    def display_text(self) -> str:
        return _display_texts[self.name]

    @property
    def is_pil_image(self) -> bool:
        raise NotImplementedError()

    @property
    def n_dimensions(self) -> int:
        raise NotImplementedError()

    @property
    def dataset_class(self) -> type:
        raise NotImplementedError()

    def available_image_resolutions(self) -> tuple[int, ...]:
        return get_available_resolutions(self)

    def available_image_sizes(self) -> tuple[tuple[int, ...], ...]:
        ndim = self.n_dimensions
        return tuple((size,) * ndim for size in get_available_resolutions(self))

    @property
    def description(self) -> str:
        return self.info()['description']


_display_texts = {
    'path_mnist': 'PathMNIST',
    'chest_mnist': 'ChestMNIST',
    'derma_mnist': 'DermaMNIST',
    'oct_mnist': 'OCTMNIST',
    'pneumonia_mnist': 'PneumoniaMNIST',
    'retina_mnist': 'RetinaMNIST',
    'breast_mnist': 'BreastMNIST',
    'blood_mnist': 'BloodMNIST',
    'tissue_mnist': 'TissueMNIST',
    'organ_a_mnist': 'OrganAMNIST',
    'organ_c_mnist': 'OrganCMNIST',
    'organ_s_mnist': 'OrganSMNIST',
    'organ_mnist3d': 'OrganMNIST3D',
    'nodule_mnist3d': 'NoduleMNIST3D',
    'adrenal_mnist3d': 'AdrenalMNIST3D',
    'fracture_mnist3d': 'FractureMNIST3D',
    'vessel_mnist3d': 'VesselMNIST3D',
    'synapse_mnist3d': 'SynapseMNIST3D',
}


class MedMNIST2D(_MedMNISTEnumMixin):
    """
    Allows easy referencing the medmnist 2D datasets.

    Documentation texts taken from :const:`medmnist.INFO`.
    """

    path_mnist = medmnist.PathMNIST
    """
    Task: Multi-class classification.    
    Channels: 3.
    
    The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a
    dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological
    images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is
    comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of
    3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The
    CRC-VAL-HE-7K is treated as the test set.
    """

    chest_mnist = medmnist.ChestMNIST
    """
    Task: Multi-label binary classification.    
    Channels: 1.
    
    The ChestMNIST is based on the NIH-ChestXray14 dataset, a dataset comprising 112,120 frontal-view X-Ray images of
    30,805 unique patients with the text-mined 14 disease labels, which could be formulized as a multi-label
    binary-class classification task. We use the official data split, and resize the source images of 1×1024×1024 into
    1×28×28.
    """

    derma_mnist = medmnist.DermaMNIST
    """
    Task: Multi-class classification.    
    Channels: 3.
    
    The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented
     skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized
     as a multi-class classification task. We split the images into training, validation and test set with a ratio of
     7:1:2. The source images of 3×600×450 are resized into 3×28×28.
    """

    oct_mnist = medmnist.OCTMNIST
    """
    Task: Multi-class classification.    
    Channels: 1.
    
    The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal
    diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split
     the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as
     the test set. The source images are gray-scale, and their sizes are (384−1,536)×(277−512). We center-crop the
     images and resize them into 1×28×28.
    """

    pneumonia_mnist = medmnist.PneumoniaMNIST
    """
    Task: Binary classification.    
    Channels: 1.
    
    The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class
    classification of pneumonia against normal. We split the source training set with a ratio of 9:1 into training and
    validation set and use its source validation set as the test set. The source images are gray-scale, and their sizes
    are (384−2,916)×(127−2,713). We center-crop the images and resize them into 1×28×28.
    """

    retina_mnist = medmnist.RetinaMNIST
    """
    Task: Ordinal regression.    
    Channels: 3.
    
    The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images.
    The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training
    set with a ratio of 9:1 into training and validation set, and use the source validation set as the test set.
    The source images of 3×1,736×1,824 are center-cropped and resized into 3×28×28.
    """

    breast_mnist = medmnist.BreastMNIST
    """
    Task: Binary classification.
    Channels: 1.
    
    The BreastMNIST is based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal,
    benign, and malignant. As we use low-resolution images, we simplify the task into binary classification by
    combining normal and benign as positive and classifying them against malignant as negative. We split the source
    dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1×500×500 are resized
    into 1×28×28.
    """

    blood_mnist = medmnist.BloodMNIST
    """
    Task: Multi-class classification.
    Channels: 3.
    
    The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without
    infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood
    collection. It contains a total of 17,092 images and is organized into 8 classes. We split the source dataset with
    a ratio of 7:1:2 into training, validation and test set. The source images with resolution 3×360×363 pixels are
    center-cropped into 3×200×200, and then resized into 3×28×28.
    """

    tissue_mnist = medmnist.TissueMNIST
    """
    Task: Multi-class classification.
    Channels: 1.
    
    We use the BBBC051, available from the Broad Bioimage Benchmark Collection. The dataset contains 236,386 human
    kidney cortex cells, segmented from 3 reference tissue specimens and organized into 8 categories. We split the
    source dataset with a ratio of 7:1:2 into training, validation and test set. Each gray-scale image is 32×32×7
    pixels, where 7 denotes 7 slices. We take maximum values across the slices and resize them into 28×28 gray-scale
    images.
    """

    organ_a_mnist = medmnist.OrganAMNIST
    """
    Task: Multi-class classification.
    Channels: 1.
    
    The OrganAMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS).
    It is renamed from OrganMNIST_Axial (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body
    organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into
    gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in axial
    views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs.
    115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT
    scans from the source test set are treated as the test set.
    """

    organ_c_mnist = medmnist.OrganCMNIST
    """
    Task: Multi-class classification.
    Channels: 1.
    
    The OrganCMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS).
    It is renamed from OrganMNIST_Coronal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body
    organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into
    gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in coronal
    views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115
    and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans
    from the source test set are treated as the test set.
    """

    organ_s_mnist = medmnist.OrganSMNIST
    """
    Task: Multi-class classification.
    Channels: 1.
    
    The OrganSMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS).
    It is renamed from OrganMNIST_Sagittal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body
    organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into
    gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in sagittal
    views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115
    and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans
    from the source test set are treated as the test set.
    """

    @property
    def size_28x28(self) -> MnistSubset:
        return MnistSubset(self.value, size=28)

    @property
    def size_64x64(self) -> MnistSubset:
        return MnistSubset(self.value, size=64)

    @property
    def size_128x128(self) -> MnistSubset:
        return MnistSubset(self.value, size=128)

    @property
    def size_224x224(self) -> MnistSubset:
        return MnistSubset(self.value, size=224)

    @property
    def dataset_name(self) -> str:
        return self.value.flag

    @property
    def is_pil_image(self) -> bool:
        return True

    @property
    def n_dimensions(self) -> int:
        return 2

    @property
    def dataset_class(self) -> type:
        return self.value


class MedMNIST3D(_MedMNISTEnumMixin):
    """
    Allows easy referencing the medmnist 2D datasets.

    Documentation texts taken from :const:`medmnist.INFO`.
    """

    organ_mnist3d = medmnist.OrganMNIST3D
    """
    Task: Multi-class classification.
    Channels: 1.
    
    The source of the OrganMNIST3D is the same as that of the Organ{A,C,S}MNIST. Instead of 2D images, we directly use
    the 3D bounding boxes and process the images into 28×28×28 to perform multi-class classification of 11 body organs.
    The same 115 and 16 CT scans as the Organ{A,C,S}MNIST from the source training set are used as training and
    validation set, respectively, and the same 70 CT scans as the Organ{A,C,S}MNIST from the source test set are
    treated as the test set.
    """

    nodule_mnist3d = medmnist.NoduleMNIST3D
    """
    Task: Binary classification.
    Channels: 1.
    
    The NoduleMNIST3D is based on the LIDC-IDRI, a large public lung nodule dataset, containing images from thoracic CT
    scans. The dataset is designed for both lung nodule segmentation and 5-level malignancy classification task. To
    perform binary classification, we categorize cases with malignancy level 1/2 into negative class and 4/5 into
    positive class, ignoring the cases with malignancy level 3. We split the source dataset with a ratio of 7:1:2 into
    training, validation and test set, and center-crop the spatially normalized images (with a spacing of 1mm×1mm×1mm)
    into 28×28×28.
    """

    adrenal_mnist3d = medmnist.AdrenalMNIST3D
    """
    Task: Binary classification.
    Channels: 1.
    
    The AdrenalMNIST3D is a new 3D shape classification dataset, consisting of shape masks from 1,584 left and right
    adrenal glands (i.e., 792 patients). Collected from Zhongshan Hospital Affiliated to Fudan University, each 3D shape
    of adrenal gland is annotated by an expert endocrinologist using abdominal computed tomography (CT), together with
    a binary classification label of normal adrenal gland or adrenal mass. Considering patient privacy, we do not
    provide the source CT scans, but the real 3D shapes of adrenal glands and their classification labels. We calculate
    the center of adrenal and resize the center-cropped 64mm×64mm×64mm volume into 28×28×28. The dataset is randomly
    split into training/validation/test set of 1,188/98/298 on a patient level.
    """

    fracture_mnist3d = medmnist.FractureMNIST3D
    """
    Task: Multi-class classification.
    Channels: 1.
    
    The FractureMNIST3D is based on the RibFrac Dataset, containing around 5,000 rib fractures from 660 computed
    tomography 153 (CT) scans. The dataset organizes detected rib fractures into 4 clinical categories (i.e., buckle,
    nondisplaced, displaced, and segmental rib fractures). As we use low-resolution images, we disregard segmental rib
    fractures and classify 3 types of rib fractures (i.e., buckle, nondisplaced, and displaced). For each annotated
    fracture area, we calculate its center and resize the center-cropped 64mm×64mm×64mm image into 28×28×28.
    The official split of training, validation and test set is used.
    """

    vessel_mnist3d = medmnist.VesselMNIST3D
    """
    Task: Binary classification.
    Channels: 1.
    
    The VesselMNIST3D is based on an open-access 3D intracranial aneurysm dataset, IntrA, containing 103 3D models
    (meshes) of entire brain vessels collected by reconstructing MRA images. 1,694 healthy vessel segments and 215
    aneurysm segments are generated automatically from the complete models. We fix the non-watertight mesh with
    PyMeshFix and voxelize the watertight mesh with trimesh into 28×28×28 voxels. We split the source dataset with a
    ratio of 7:1:2 into training, validation and test set.
    """

    synapse_mnist3d = medmnist.SynapseMNIST3D
    """
    Task: Binary classification.
    Channels: 1.
    
    The SynapseMNIST3D is a new 3D volume dataset to classify whether a synapse is excitatory or inhibitory. It uses a
    3D image volume of an adult rat acquired by a multi-beam scanning electron microscope. The original data is of the
    size 100×100×100um^3 and the resolution 8×8×30nm^3, where a (30um)^3 sub-volume was used in the MitoEM dataset with
    dense 3D mitochondria instance segmentation labels. Three neuroscience experts segment a pyramidal neuron within the
    whole volume and proofread all the synapses on this neuron with excitatory/inhibitory labels. For each labeled
    synaptic location, we crop a 3D volume of 1024×1024×1024nm^3 and resize it into 28×28×28 voxels. Finally, the
    dataset is randomly split with a ratio of 7:1:2 into training, validation and test set.
    """

    @property
    def size_28x28x28(self) -> MnistSubset:
        return MnistSubset(self.value, size=28)

    @property
    def size_64x64x64(self) -> MnistSubset:
        return MnistSubset(self.value, size=64)

    @property
    def is_pil_image(self) -> bool:
        return False

    @property
    def n_dimensions(self) -> int:
        return 3

    @property
    def dataset_name(self) -> str:
        return self.value.flag

    @property
    def dataset_class(self) -> type:
        return self.value


def get_available_resolutions(dataset: Union[MedMNISTDataSource, Type[medmnist.dataset.MedMNIST]]) -> tuple[int, ...]:
    """
    Returns the image dimensions in which the indicated dataset is available. The image dimensions are equal across each
    coordinate axis.

    :param dataset: The dataset.
    :return: A tuple of available sizes.
    """

    if isinstance(dataset, MedMNISTDataSource):
        dataset = dataset.dataset_class
    elif not issubclass(dataset, type):
        raise ValueError('Type expected as dataset.')

    sizes = getattr(dataset, 'available_sizes', None)

    if sizes is None:
        raise ValueError(f'Unsupported type {dataset}.')

    return sizes


def get_dataset_dimension(
        dataset: Union[MedMNISTDataSource, medmnist.dataset.MedMNIST, Type[medmnist.dataset.MedMNIST]]) -> int:
    """
    Returns the number of dimensions of the indicated dataset.

    :param dataset: The dataset.
    :return: The number of dimensions.
    """
    if isinstance(dataset, MedMNISTDataSource):
        dataset = dataset.dataset_class
    elif not issubclass(dataset, type):
        dataset = type(dataset)

    if issubclass(dataset, medmnist.dataset.MedMNIST3D):
        return 3
    elif issubclass(dataset, medmnist.dataset.MedMNIST2D):
        return 2
    else:
        raise ValueError('Unsupported dataset type.')


def initialize_mnist(datasource: MnistSubset, root_directory: str,
                     *splits: Union[str, DatasetSplit],
                     convert_to_numpy: bool = True,
                     channel_last: bool = False,
                     download_to_root_directory: bool = True,
                     unpack_singleton: bool = True) -> \
        Union[torch.utils.data.Dataset, tuple[torch.utils.data.Dataset, ...]]:
    """
    Creates a :class:`torch.utils.data.Dataset` for the indicated data source.

    :param datasource: The dataset subset to load.
    :param splits: The portion of the subset to take. Allowed strings: 'train', 'val', 'test'.
    :param root_directory: The root directory in which to search for the dataset and to download to, if parameter
        ``download_to_root_directory`` is set to :const:`True`.
    :param convert_to_numpy: Set to ``True`` to ensure that numpy arrays are returned. 2D datasets may otherwise be
        returned as a :class:`PIL.Image.Image`.
    :param channel_last: Set to :const:`True` to ensure that the format is HxW(xD)xC and not CxHxW(xD).
    :param download_to_root_directory: If :const:`True`, downloads the dataset to the indicated root directory if the
        indicated dataset is not found.
    :param unpack_singleton: Set to ``False`` in case single splits should be kept as a tuple (e.g., supplying only
        'val' as a split will return a tuple with a single entry. If ``True``, instead of the tuple, the sole value in
        it would be unpacked and returned).
    :return: If :paramref:`unpack_singleton` is ``True`` and :paramref:`splits` contains a single value, this will
        return a dataset, otherwise this will return a tuple of datasets.
    """

    def _transform_image(image: Union[np.ndarray, PIL.Image.Image]) -> Union[np.ndarray, PIL.Image.Image]:
        is_channel_last = False

        if convert_to_numpy and not isinstance(image, np.ndarray):
            image = np.asarray(image)

            # convert int8 image to float
            if not np.issubdtype(image.dtype, np.floating):
                image = image / 255.

                # some images give a squeezed tensor, but we want the channels to be in a separate axis
                if image.ndim != 3:
                    image = np.expand_dims(image, -1)

                # we get a channel-last array --> we may need to convert that in the subsequent step
                is_channel_last = True

        if channel_last and not is_channel_last:
            # 3D images are stored in channel-first order, not in channel-last order
            if isinstance(image, np.ndarray):
                image = np.moveaxis(image, 0, -1)
        elif not channel_last and is_channel_last:
            if isinstance(image, np.ndarray):
                image = np.moveaxis(image, -1, 0)

        return image

    # def _transform_target(target: np.ndarray) -> np.ndarray:
    #     return target

    splits = tuple(DatasetSplit(split).value for split in splits)

    datasets = tuple(datasource.mnist_type(**datasource.parameters,
                                           split=split,
                                           download=download_to_root_directory,
                                           root=root_directory,
                                           transform=_transform_image) for split in splits)

    if unpack_singleton and len(datasets) == 1:
        return datasets[0]
    else:
        return datasets
