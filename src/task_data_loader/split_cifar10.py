from typing import List, Optional, Callable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TaskSpecificSplitCIFAR10(Dataset):
    classes = {
        0: "plane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    def __init__(self, features: np.ndarray, targets: List[int], task_id: int, transform: Optional[Callable] = None):
        self.features = features
        self.targets = targets
        self.task_id = task_id
        self.transform = transform
        self.target_classes_ID = set(targets)
        self.target_classes_dict = {
            idx: class_str for idx, class_str in self.classes.items() if idx in self.target_classes_ID
        }
        self.internal_target_def = {
            external_id: internal_id for internal_id, external_id in enumerate(self.target_classes_ID)
        }
        self.reverse_internal_target = {
            internal_id: self.target_classes_dict[external_id]
            for external_id, internal_id in self.internal_target_def.items()
        }

    def __getitem__(self, item: int):
        img, target = self.features[item], self.targets[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.internal_target_def[target]

    def __len__(self) -> int:
        return len(self.features)

    def __str__(self) -> str:
        return f"Task ID: {self.task_id}\nTarget Classes: {self.target_classes_dict}"


cifar10_basic_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
