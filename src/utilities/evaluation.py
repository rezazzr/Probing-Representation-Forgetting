from typing import Dict, Sequence, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from models.cifar10 import TaskBasedNets
from models.imagenet_based_models import VGG16
from task_data_loader.scenarios import Scenario
from task_data_loader.split_cifar10 import TaskSpecificSplitCIFAR10
from utilities.metrics import PredictionBasedMetric, RepresentationBasedMetric
from utilities.utils import gpu_information_summary, merge, to_numpy


class PredictionBasedEvaluator:
    def __init__(self, metrics: Sequence[PredictionBasedMetric], batch_size: int = 32, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        n_gpu, self.device = gpu_information_summary(show=False)
        self.metrics = metrics

    def eval_all_tasks(self, model: Union[TaskBasedNets, VGG16], data_stream: Scenario):
        task_evaluation = dict()
        for task in data_stream.tasks:
            task_eval_metrics = self.eval_one_task(
                model=model, task=task.test, task_id=task.id, nb_classes=task.nb_classes
            )
            for metric_name, metric_dict_value in task_eval_metrics.items():
                task_eval_metrics[metric_name] = {
                    f"task_{task.id}_{key}": value for key, value in metric_dict_value.items()
                }
            task_evaluation = merge(task_evaluation, task_eval_metrics)
        return task_evaluation

    def eval_one_task(
        self,
        model: Union[TaskBasedNets, VGG16],
        task: Union[Dataset, TaskSpecificSplitCIFAR10],
        task_id: Optional[str] = None,
        nb_classes: int = -1,
    ) -> Dict[str, Dict[str, float]]:
        self.before_eval_one_task(task=task, nb_classes=nb_classes)
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            task, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, evaluation_targets = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                evaluation_targets = evaluation_targets.numpy()
                logits = to_numpy(
                    model(evaluation_features, task_id) if task_id is not None else model(evaluation_features)
                )
                self.eval_one_batch(logits=logits, targets=evaluation_targets)

        return self.compute_eval_one_task()

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray) -> None:
        for metric in self.metrics:
            metric.eval_one_batch(logits=logits, targets=targets)

    def compute_eval_one_task(self) -> Dict[str, Dict[str, float]]:
        metric_evaluation = dict()
        for metric in self.metrics:
            metric_evaluation[type(metric).__name__] = metric.compute_metric()

        return metric_evaluation

    def before_eval_one_task(self, task: Union[Dataset, TaskSpecificSplitCIFAR10], nb_classes: int = -1) -> None:
        for metric in self.metrics:
            metric.initialize_metric(task=task, nb_classes=nb_classes)


class RepresentationBasedEvaluator:
    def __init__(self, metrics: Sequence[RepresentationBasedMetric], batch_size: int = 32, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        _, self.device = gpu_information_summary(show=False)
        self.metrics = metrics

    def record_original_representations(
        self, model: TaskBasedNets, task: Union[Dataset, TaskSpecificSplitCIFAR10], task_id: str = "."
    ) -> None:
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            task, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, _ = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                block_reps = model.block_forward(evaluation_features, task_id)
                if batch_number == 0:
                    self.initialize_memory(representation_blocks=block_reps, is_old_blocks=True)
                else:
                    self.aggregate_batches(representation_blocks=block_reps, is_old_blocks=True)

    def record_updated_representations(
        self, model: TaskBasedNets, task: Union[Dataset, TaskSpecificSplitCIFAR10], task_id: str = "."
    ) -> None:
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            task, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, _ = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                block_reps = model.block_forward(evaluation_features, task_id)
                if batch_number == 0:
                    self.initialize_memory(representation_blocks=block_reps, is_old_blocks=False)
                else:
                    self.aggregate_batches(representation_blocks=block_reps, is_old_blocks=False)

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        metric_evaluation = dict()
        for metric in self.metrics:
            metric_evaluation[type(metric).__name__] = metric.compute_metric()
        return metric_evaluation

    def initialize_memory(self, representation_blocks: Dict[str, torch.Tensor], is_old_blocks: bool = True):
        for metric in self.metrics:
            metric.initialize_memory(representation_blocks=representation_blocks, is_old_blocks=is_old_blocks)

    def aggregate_batches(self, representation_blocks: Dict[str, torch.Tensor], is_old_blocks: bool = True):
        for metric in self.metrics:
            metric.aggregate_batches(representation_blocks=representation_blocks, is_old_blocks=is_old_blocks)
