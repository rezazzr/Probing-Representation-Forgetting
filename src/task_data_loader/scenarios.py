import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List, Dict, Union

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from task_data_loader.imagenet import train_transform, valid_transform
from task_data_loader.split_cifar10 import TaskSpecificSplitCIFAR10, cifar10_basic_transform
from .cub import Cub2011
from .flowers import Flowers102
from .scenes import Scenes


@dataclass
class TaskConfig:
    train: Dataset
    test: Dataset
    id: str
    nb_classes: int


class Scenario(ABC):
    def __init__(self, root: str, transforms: Optional[Union[List[Callable], Callable]]):
        self.root = root
        self.transforms = transforms

    @property
    @abstractmethod
    def tasks(self) -> List[TaskConfig]:
        pass


class DuplicatedHalfCIFAR10(Scenario):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str = "data",
        task1_classes: Tuple[int, ...] = (0, 1, 2, 3, 4),
        transforms: Optional[Callable] = cifar10_basic_transform,
    ):
        super().__init__(root, transforms)
        self.task1_classes = task1_classes
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset_into_two_tasks(
            features=train_dataset["features"], targets=train_dataset["targets"], task1_classes=self.task1_classes
        )
        split_test_tasks = self._split_dataset_into_two_tasks(
            features=test_dataset["features"], targets=test_dataset["targets"], task1_classes=self.task1_classes
        )

        self.task1 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task1"]["features"],
                targets=split_train_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task1"]["features"],
                targets=split_test_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            id="1",
            nb_classes=5,
        )

        self.task2 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task1"]["features"],
                targets=split_train_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task1"]["features"],
                targets=split_test_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            id="2",
            nb_classes=5,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.task1, self.task2]

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset_into_two_tasks(self, features: np.ndarray, targets: List[int], task1_classes: Tuple[int, ...]):
        task1_features = []
        task1_targets = []
        task2_features = []
        task2_targets = []

        for feature, target in zip(features, targets):
            if target in task1_classes:
                task1_features.append(feature)
                task1_targets.append(target)
            else:
                task2_features.append(feature)
                task2_targets.append(target)

        task1_features = np.asarray(task1_features)
        task2_features = np.asarray(task2_features)

        return {
            "task1": {"features": task1_features, "targets": task1_targets},
            "task2": {"features": task2_features, "targets": task2_targets},
        }


class SplitCIFAR10(Scenario):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str = "data",
        task1_classes: Tuple[int, ...] = (0, 1, 2, 3, 4),
        transforms: Optional[Callable] = cifar10_basic_transform,
    ):
        super().__init__(root, transforms)
        self.task1_classes = task1_classes
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset_into_two_tasks(
            features=train_dataset["features"], targets=train_dataset["targets"], task1_classes=self.task1_classes
        )
        split_test_tasks = self._split_dataset_into_two_tasks(
            features=test_dataset["features"], targets=test_dataset["targets"], task1_classes=self.task1_classes
        )

        self.task1 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task1"]["features"],
                targets=split_train_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task1"]["features"],
                targets=split_test_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            id="1",
            nb_classes=5,
        )

        self.task2 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task2"]["features"],
                targets=split_train_tasks["task2"]["targets"],
                task_id=2,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task2"]["features"],
                targets=split_test_tasks["task2"]["targets"],
                task_id=2,
                transform=self.transforms,
            ),
            id="2",
            nb_classes=5,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.task1, self.task2]

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset_into_two_tasks(self, features: np.ndarray, targets: List[int], task1_classes: Tuple[int, ...]):
        task1_features = []
        task1_targets = []
        task2_features = []
        task2_targets = []

        for feature, target in zip(features, targets):
            if target in task1_classes:
                task1_features.append(feature)
                task1_targets.append(target)
            else:
                task2_features.append(feature)
                task2_targets.append(target)

        task1_features = np.asarray(task1_features)
        task2_features = np.asarray(task2_features)

        return {
            "task1": {"features": task1_features, "targets": task1_targets},
            "task2": {"features": task2_features, "targets": task2_targets},
        }


class ImageNetScenario(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root=self.root, split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root=self.root, split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet]


class ScenesScenario(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.scenes]


class ImageNet2CUB(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root=self.root, split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root=self.root, split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )
        self.cub = TaskConfig(
            train=Cub2011(root=self.root, train=True, transform=self.transforms[0]),
            test=Cub2011(root=self.root, train=False, transform=self.transforms[1]),
            id="cub",
            nb_classes=200,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.cub]


class ImageNet2Scenes(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root=self.root, split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root=self.root, split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.scenes]


class ImageNet2Scenes2CUB(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root=self.root, split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root=self.root, split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

        self.cub = TaskConfig(
            train=Cub2011(root=self.root, train=True, transform=self.transforms[0]),
            test=Cub2011(root=self.root, train=False, transform=self.transforms[1]),
            id="cub",
            nb_classes=200,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.scenes, self.cub]


class ImageNet2Flowers(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root=self.root, split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root=self.root, split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )
        self.flowers = TaskConfig(
            train=Flowers102(root=self.root, train=True, transform=self.transforms[0]),
            test=Flowers102(root=self.root, train=False, transform=self.transforms[1]),
            id="flowers",
            nb_classes=102,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.flowers]


class ImageNet2Scenes2CUB2Flowers(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root=self.root, split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root=self.root, split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

        self.cub = TaskConfig(
            train=Cub2011(root=self.root, train=True, transform=self.transforms[0]),
            test=Cub2011(root=self.root, train=False, transform=self.transforms[1]),
            id="cub",
            nb_classes=200,
        )
        self.flowers = TaskConfig(
            train=Flowers102(root=self.root, train=True, transform=self.transforms[0]),
            test=Flowers102(root=self.root, train=False, transform=self.transforms[1]),
            id="flowers",
            nb_classes=102,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.scenes, self.cub, self.flowers]
