from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
from torch.nn import modules, CrossEntropyLoss
from torch.utils.data import Dataset

from task_data_loader.split_cifar10 import TaskSpecificSplitCIFAR10
from utilities.cka import TorchCKA
from utilities.utils import gpu_information_summary


class PredictionBasedMetric(ABC):
    @abstractmethod
    def initialize_metric(self, task: Union[Dataset, TaskSpecificSplitCIFAR10], **kwargs):
        pass

    @abstractmethod
    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray):
        pass

    @abstractmethod
    def compute_metric(self) -> Dict[str, float]:
        pass


class RepresentationBasedMetric(ABC):
    @abstractmethod
    def __init__(self):
        self.old_representation_blocks: Dict[str, torch.Tensor] = dict()
        self.new_representation_blocks: Dict[str, torch.Tensor] = dict()

    def initialize_memory(self, representation_blocks: Dict[str, torch.Tensor], is_old_blocks: bool = True):
        if is_old_blocks:
            self.old_representation_blocks = representation_blocks
        else:
            self.new_representation_blocks = representation_blocks

    def aggregate_batches(self, representation_blocks: Dict[str, torch.Tensor], is_old_blocks: bool = True):
        if is_old_blocks:
            for block_name, block_value in representation_blocks.items():
                self.old_representation_blocks[block_name] = torch.cat(
                    (self.old_representation_blocks[block_name], block_value), axis=0
                )
        else:
            for block_name, block_value in representation_blocks.items():
                self.new_representation_blocks[block_name] = torch.cat(
                    (self.new_representation_blocks[block_name], block_value), axis=0
                )

    @abstractmethod
    def compute_metric(self) -> Dict[str, float]:
        pass


class Accuracy(PredictionBasedMetric):
    def __init__(self):
        self.correct_predictions = dict()
        self.total_predictions = dict()

    def initialize_metric(self, task: Union[Dataset, TaskSpecificSplitCIFAR10], **kwargs):
        self.correct_predictions = {i: 0 for i in range(kwargs["nb_classes"])}
        self.total_predictions = {i: 0 for i in range(kwargs["nb_classes"])}

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray):
        predictions = self._logits_to_predictions(logits=logits)
        for target, prediction in zip(targets, predictions):
            if target == prediction:
                self.correct_predictions[target] += 1
            self.total_predictions[target] += 1

    def compute_metric(self) -> Dict[str, float]:
        return {"overall": (100.0 * sum(self.correct_predictions.values())) / sum(self.total_predictions.values())}

    @staticmethod
    def _logits_to_predictions(logits: np.ndarray) -> np.ndarray:
        return np.argmax(logits, axis=1)


class CKA(RepresentationBasedMetric):
    def __init__(self, use_cuda=False):
        super(CKA, self).__init__()
        _, self.device = gpu_information_summary(show=False)
        self.cuda_cka = TorchCKA(device=self.device)

    def compute_metric(self) -> Dict[str, float]:
        cka_results = dict()
        for block_name, old_block_value in self.old_representation_blocks.items():
            cka_results[block_name] = self.cuda_cka.linear_CKA(
                X=old_block_value, Y=self.new_representation_blocks[block_name]
            )
        return cka_results


# TODO: Needs to be adapted with new changes in the representation based metrics
class L2(RepresentationBasedMetric):
    def __init__(self, normalize: bool = True):
        super(L2, self).__init__()
        self.normalize = normalize

    def compute_metric(self) -> Dict[str, float]:
        l2_results = dict()
        for block_name, old_block_value in self.old_representation_blocks.items():
            new_block_value = self.new_representation_blocks[block_name]
            if self.normalize:
                old_block_value = self._normalize(matrix=old_block_value)
                new_block_value = self._normalize(matrix=new_block_value)
            l2_results[block_name] = np.linalg.norm(old_block_value - new_block_value)

        return l2_results

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.linalg.norm(matrix, axis=-1)[:, np.newaxis]


class Loss(PredictionBasedMetric):
    def __init__(self, criterion: modules.loss._Loss = CrossEntropyLoss()):
        self.criterion = criterion
        self.summed_loss = 0.0
        self.batches_processed = 0

    def initialize_metric(self, task: Union[Dataset, TaskSpecificSplitCIFAR10], **kwargs):
        self.summed_loss = 0.0
        self.batches_processed = 0

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray):
        loss = self.criterion(torch.from_numpy(logits), torch.from_numpy(targets))
        self.summed_loss += loss.item()
        self.batches_processed += 1

    def compute_metric(self) -> Dict[str, float]:
        return {"eval_loss": self.summed_loss / self.batches_processed}
