from dataclasses import dataclass
from typing import Optional, List

import torch

from utilities.evaluation import PredictionBasedEvaluator, RepresentationBasedEvaluator
from utilities.loggers import Loggers
from utilities.utils import EarlyStoppingConfig


@dataclass
class TrainingConfig:
    prediction_evaluator: Optional[PredictionBasedEvaluator] = None
    representation_evaluator: Optional[RepresentationBasedEvaluator] = None
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: Optional[torch.optim.Optimizer] = None
    seed_value: int = 1609
    nb_epochs: int = 2
    num_workers: int = 0
    batch_size: int = 32
    logging_step: int = 1
    loggers: Optional[List[Loggers]] = None
    early_stopping_config: Optional[EarlyStoppingConfig] = None
    is_probe: bool = False
    verbose: bool = True
    save_progress: bool = False
    saving_dir: Optional[str] = "model_zoo"
    strategy: str = "FineTuning"
    probe_caller: Optional[str] = None
    use_scheduler: bool = True
    nb_warmup_steps: int = -1
    learning_rate: float = 1e-5
    max_steps: int = -1
    use_different_seed: bool = False
    experiment_name: Optional[str] = None
    max_grad_norm: float = 1.0
    use_sup_con: bool = False
    nb_epochs_supcon: int = -1


@dataclass
class OneShotConfig:
    prediction_evaluator: Optional[PredictionBasedEvaluator] = None
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: Optional[torch.optim.Optimizer] = None
    seed_value: int = 1609
    nb_epochs: int = 2
    nb_epochs_supcon: int = -1
    num_workers: int = 0
    batch_size: int = 32
    logging_step: int = 1
    loggers: Optional[List[Loggers]] = None
    early_stopping_config: Optional[EarlyStoppingConfig] = None
    verbose: bool = True
    save_progress: bool = False
    progress_history: int = 1
    saving_dir: Optional[str] = "model_zoo"
    strategy: str = "FineTuning"
    use_scheduler: bool = True
    nb_warmup_steps: int = -1
    learning_rate: float = 1e-5
    max_steps: int = -1
    experiment_name: Optional[str] = None
    nb_classes: int = 1
    max_grad_norm: float = 1.0
    use_sup_con: bool = False
