import math
import os
import random
import typing
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Dict
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from prettytable import PrettyTable
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR

CIFAR10_CLASSES = {
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


def gpu_information_summary(show: bool = True) -> Tuple[int, torch.device]:
    """
    :param show: Controls whether or not to print the summary information
    :return: number of gpus and the device (CPU or GPU)
    """
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name() if n_gpu > 0 else "None"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    table.add_row(["GPU", gpu_name])
    table.add_row(["Number of GPUs", n_gpu])
    if show:
        print(table)
    return n_gpu, device


def set_seed(seed_value: int, n_gpu: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_value)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def img_show(img) -> None:
    undo_default_transformation = transforms.Compose(
        [
            transforms.Normalize((0.0, 0.0, 0.0), (1 / 0.2023, 1 / 0.1994, 1 / 0.2010)),
            transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
        ]
    )
    img = undo_default_transformation(img)
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def merge(dict1: Dict, dict2: Dict) -> Dict:
    """Return a new dictionary by merging two dictionaries recursively."""

    result = deepcopy(dict1)

    for key, value in dict2.items():
        if isinstance(value, dict):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])

    return result


def xavier_uniform_initialize(layer: Module):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    if type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)


@dataclass
class CheckPointingConfig:
    model_name: str = "NN"
    history: int = 1
    verbose: bool = True
    directory: str = "progress_checkpoints"

    @property
    def address(self) -> str:
        return os.path.join(self.directory, self.model_name)


@dataclass
class EarlyStoppingConfig:
    model_name: str = "NN"
    patience: int = 5
    verbose: bool = True
    delta: float = 0
    directory: str = "checkpoints"

    @property
    def address(self) -> str:
        return os.path.join(self.directory, self.model_name)

    @property
    def path(self) -> str:
        return os.path.join(self.address, "checkpoint.pt")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, config: EarlyStoppingConfig = EarlyStoppingConfig(), trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = config.patience
        self.verbose = config.verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = config.delta
        self.path = config.path
        self.trace_func = trace_func
        if not os.path.exists(config.address):
            os.makedirs(config.address)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def zero_like_params_dict(model: torch.nn.Module):
    """
    Create a list of (name, parameter), where parameter is initialized to zero.
    The list has as many parameters as the model, with the same size.
    :param model: a pytorch model
    """

    return [(k, torch.zeros_like(p).to(p.device)) for k, p in model.named_parameters()]


def copy_params_dict(model: torch.nn.Module, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.clone()) for k, p in model.named_parameters()]


class CheckPointManager:
    def __init__(self, config: CheckPointingConfig = CheckPointingConfig(), trace_func=print):
        self.config = config
        self.saved_history = []
        self.trace_func = trace_func
        if not os.path.exists(config.address):
            os.makedirs(config.address)

    def __call__(self, model, step, optimizer):
        path = os.path.join(self.config.address, f"checkpoint_{step}.pt")

        if self.config.verbose:
            self.trace_func(f"Saving model at\n {path}")

        if len(self.saved_history) >= self.config.history:
            os.remove(self.saved_history.pop(0))

        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{path}",
        )
        self.saved_history.append(path)


def safely_load_state_dict(checkpoint_path: str) -> typing.OrderedDict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path)["model_state_dict"]
    final_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            final_state_dict[key[7:]] = value
        else:
            final_state_dict[key] = value
    return final_state_dict


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
