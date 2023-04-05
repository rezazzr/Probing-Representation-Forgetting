import copy
import os
import time
from dataclasses import replace
from typing import Optional, List
from typing import Tuple, Any, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from models.cifar10 import TaskBasedNets
from models.imagenet_based_models import PredictionLayerConfig, VGG16, MiniResNet, MiniResNetSupCon
from models.probes import LinearProbeImageNet, LinearProbeCIFAR10
from strategy.ewc import EWCStrategy, EWCStrategy8000, EWCStrategy500
from strategy.lwf import LwFStrategy
from strategy.supcon import SupConLoss
from task_data_loader.scenarios import Scenario
from task_data_loader.split_cifar10 import TaskSpecificSplitCIFAR10
from utilities.configs import TrainingConfig, OneShotConfig
from utilities.evaluation import PredictionBasedEvaluator
from utilities.loggers import TensorboardLogger, IOLogger
from utilities.metrics import Loss
from utilities.utils import (
    gpu_information_summary,
    EarlyStopping,
    cosine_with_hard_restarts_schedule_with_warmup,
    xavier_uniform_initialize,
    set_seed,
    CheckPointManager,
    CheckPointingConfig,
    TwoCropTransform,
)
import random
from dataclasses import asdict
from tqdm.auto import tqdm


class ProbeEvaluator:
    def __init__(
        self,
        blocks_to_prob: List[str],
        data_stream: Scenario,
        half_precision: bool = False,
        prediction_layers: Optional[List[PredictionLayerConfig]] = None,
        training_configs: TrainingConfig = TrainingConfig(),
    ):
        self.half_precision = half_precision
        self.training_configs = training_configs
        self.prediction_layers = prediction_layers
        self.data_stream = data_stream
        self.blocks_to_prob = blocks_to_prob

    def probe(self, model: Union[TaskBasedNets, VGG16], probe_caller: str):
        model = copy.deepcopy(model)
        block_probe_results = dict()
        for block in self.blocks_to_prob:
            block_probe_results[block] = self.probe_one_block(block=block, model=model, probe_caller=probe_caller)

        return block_probe_results

    def probe_one_block(self, block: str, model: Union[TaskBasedNets, VGG16], probe_caller: str):
        # Create probing model
        if isinstance(model, TaskBasedNets):
            probing_model = LinearProbeCIFAR10(under_investigation_model=model, intended_block=block)
        elif isinstance(model, VGG16) or isinstance(model, MiniResNet) or isinstance(model, MiniResNetSupCon):
            probing_model = LinearProbeImageNet(
                under_investigation_model=model,
                intended_block=block,
                prediction_layers=self.prediction_layers,
                half_precision=self.half_precision,
            )
        else:
            raise TypeError("Model is not of a type supported by the probe.")

        trainer = ModelCoach(
            model=probing_model,
            data_stream=self.data_stream,
            config=replace(self.training_configs, probe_caller=probe_caller),
        )
        probe_results = trainer.train()
        return probe_results


class ModelCoach:
    def __init__(
        self,
        model: Union[TaskBasedNets, LinearProbeCIFAR10, LinearProbeImageNet],
        data_stream: Scenario,
        config: TrainingConfig = TrainingConfig(),
    ):
        self.data_stream = data_stream
        self.config = config

        self.hparam_training = asdict(self.config)
        keys_to_remove = [
            "prediction_evaluator",
            "representation_evaluator",
            "criterion",
            "optimizer",
            "loggers",
            "early_stopping_config",
        ]
        for key in keys_to_remove:
            self.hparam_training.pop(key)

        if self.config.save_progress:
            if not os.path.exists(self.config.saving_dir):
                os.makedirs(self.config.saving_dir)

        if self.config.early_stopping_config is not None:
            self.config.early_stopping_config.model_name = type(model).__name__

        self.n_gpu, self.device = gpu_information_summary(show=self.config.verbose)
        self.model = model
        self.model.to(self.device)
        # TODO: Add multi-GPU support
        # if self.n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        if self.config.optimizer is None:
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
            self.config.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=5e-4,
                amsgrad=True,
            )

        self.global_step = 0
        if self.config.loggers is None:
            self.init_loggers()
        self.scheduler = None
        self.cl_strategy = None

        self.hparam_metrics = dict()
        self.running_loss = 0.0

    def train(self):
        probe_return = dict()
        global_training_step = 0
        if self.config.strategy == "LwF":
            self.cl_strategy = LwFStrategy(scenario=type(self.data_stream).__name__)
        if self.config.strategy.startswith("EWC"):
            if self.config.strategy == "EWC8000":
                self.cl_strategy = EWCStrategy8000(scenario=type(self.data_stream).__name__)
            elif self.config.strategy == "EWC500":
                self.cl_strategy = EWCStrategy500(scenario=type(self.data_stream).__name__)
            else:
                raise Exception("The EWC strategy is not supported", self.config.strategy)
        # for each task call train_tasks
        for task in self.data_stream.tasks:
            task_id = task.id

            if self.config.representation_evaluator is not None:
                self.config.representation_evaluator.record_original_representations(
                    model=self.model, task=task.test, task_id=task_id
                )

            if task_id != "imagenet" or type(self.model).__name__ == "LinearProbeImageNet":
                if self.config.use_different_seed:
                    set_seed(seed_value=2121, n_gpu=self.n_gpu)
                    self.model.apply(xavier_uniform_initialize)

                if self.config.experiment_name == "Generalized_vs_Memorized":
                    if task_id == "2":
                        random.shuffle(task.train.targets)
                elif self.config.experiment_name == "Memorized_vs_Memorized":
                    random.shuffle(task.train.targets)

                encoder_train_transform = TwoCropTransform(transform=task.train.transform)
                self.running_loss = 0.0
                if self.config.early_stopping_config is not None:
                    self.train_task(task_training=task.train, task_validation=task.test, task_id=task_id)
                else:
                    if self.config.use_sup_con:
                        task.train.transform = encoder_train_transform
                        self.train_task(task_training=task.train, task_id=task_id, encoder_only=True)
                        task.train.transform = encoder_train_transform.transform
                    else:
                        self.train_task(task_training=task.train, task_id=task_id)

                if self.config.use_sup_con:
                    self.running_loss = 0.0
                    self.train_task(task_training=task.train, task_id=task_id, encoder_only=False)

            evaluator_results = self.config.prediction_evaluator.eval_all_tasks(
                model=self.model, data_stream=self.data_stream
            )
            for metric_name, metric_value_dict in evaluator_results.items():
                self.hparam_metrics[f"hparams/{metric_name}_{task_id}"] = (
                    metric_value_dict[f"task_{task_id}_overall"]
                    if metric_name == "Accuracy"
                    else metric_value_dict[f"task_{task_id}_eval_loss"]
                )
                metric_name = f"{metric_name}/Evaluation" if metric_name == "Loss" else metric_name
                self.log(metric_name=metric_name, metric_value=metric_value_dict)

            # Saving the progress after training each task
            if self.config.save_progress:
                self.save_model(task_id=task_id)

            # Record the model for LwF distillation loss right after learning a new task
            if self.config.strategy == "LwF":
                self.cl_strategy.record_state(current_model=self.model)
            # Record the weights and their respective importance for EWC right after learning them on a task
            if self.config.strategy.startswith("EWC"):
                self.cl_strategy.record_state(
                    model=self.model,
                    criterion=self.config.criterion,
                    optimizer=self.config.optimizer,
                    dataset=task.train,
                    batch_size=self.config.batch_size,
                    current_task_id=task.id,
                )

            if self.config.representation_evaluator is not None:
                self.config.representation_evaluator.record_updated_representations(
                    model=self.model, task=task.test, task_id=task_id
                )
                representation_metrics = self.config.representation_evaluator.compute_metrics()

                for metric_name, metric_value in representation_metrics.items():
                    self.log(
                        metric_name=f"Representation/{metric_name}",
                        metric_value=metric_value,
                        global_step=global_training_step,
                    )

            global_training_step += 1
            if self.config.is_probe and self.config.prediction_evaluator is not None:
                evaluator_results = self.config.prediction_evaluator.eval_all_tasks(
                    model=self.model, data_stream=self.data_stream
                )
                probe_return[f"task_{task_id}"] = evaluator_results["Accuracy"][f"task_{task_id}_overall"]

        hparam_logging = (self.hparam_training, self.hparam_metrics)
        self.log(metric_name="hparams", metric_value=hparam_logging)

        self.terminate_logging()
        return probe_return

    def train_task(
        self,
        task_training: Union[Dataset, TaskSpecificSplitCIFAR10],
        task_validation: Optional[Union[Dataset, TaskSpecificSplitCIFAR10]] = None,
        task_id: str = ".",
        encoder_only: bool = False,
    ) -> None:
        # for each task call train_epochs
        train_loader = torch.utils.data.DataLoader(
            task_training,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        if self.config.max_steps > 0:
            self.nb_training_steps = self.config.max_steps
        else:
            self.nb_training_steps = (
                len(train_loader) * self.config.nb_epochs_supcon
                if not encoder_only and self.config.use_sup_con
                else len(train_loader) * self.config.nb_epochs
            )

        if self.config.use_scheduler:
            self.scheduler = cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.config.optimizer,
                num_warmup_steps=self.config.nb_warmup_steps,
                num_training_steps=self.nb_training_steps - self.config.nb_warmup_steps,
                num_cycles=1,
            )

        if self.config.early_stopping_config is not None:
            early_stopping = EarlyStopping(config=self.config.early_stopping_config)
            validation_monitoring = PredictionBasedEvaluator(
                metrics=[Loss()], batch_size=self.config.batch_size, num_workers=0
            )
            epoch = 0
            while not early_stopping.early_stop:
                self.train_epoch(train_loader=train_loader, epoch=epoch, task_id=task_id)
                validation_loss = validation_monitoring.eval_one_task(
                    model=self.model, task=task_validation, task_id=task_id
                )["Loss"]["eval_loss"]
                self.log(metric_name="Loss/Evaluation", metric_value=validation_loss)
                early_stopping(val_loss=validation_loss, model=self.model)
                epoch += 1

            # load the last checkpoint with the best model
            self.model.load_state_dict(torch.load(self.config.early_stopping_config.path))

        else:
            epochs = (
                self.config.nb_epochs_supcon if self.config.use_sup_con and not encoder_only else self.config.nb_epochs
            )
            for epoch in tqdm(range(epochs), desc=f"Epoch progress, {task_id}"):
                self.train_epoch(train_loader=train_loader, epoch=epoch, task_id=task_id, encoder_only=encoder_only)
                if 0 < self.config.max_steps == self.global_step:
                    break

        print("-" * 35)
        print(f"  End of Task {task_id}.")
        print("-" * 35)

    def train_epoch(
        self, train_loader: torch.utils.data.DataLoader, epoch: int, task_id: str, encoder_only: bool = False
    ) -> None:
        # for each epoch call train_iter
        for batch_number, training_instance in enumerate(train_loader):
            self.running_loss += self.train_iter(
                training_instance=training_instance, task_id=task_id, encoder_only=encoder_only
            )

            if self.global_step % 10 == 9:
                print(
                    "[Task %s, Epoch %d, Batch %5d] loss: %.3f"
                    % (task_id, epoch + 1, batch_number + 1, self.running_loss / 10)
                )
                self.log(metric_name="Loss/Training", metric_value=self.running_loss / 10)
                self.log(metric_name="LR", metric_value=self.config.optimizer.param_groups[0]["lr"])
                self.running_loss = 0.0

            if (
                (
                    self.config.logging_step > 0
                    and self.global_step % self.config.logging_step == (self.config.logging_step - 1)
                )
                and self.config.prediction_evaluator is not None
                and not encoder_only
            ):
                start_time_eval = time.time()
                evaluator_results = self.config.prediction_evaluator.eval_all_tasks(
                    model=self.model, data_stream=self.data_stream
                )
                for metric_name, metric_value_dict in evaluator_results.items():
                    self.hparam_metrics[f"hparams/{metric_name}_{task_id}"] = (
                        metric_value_dict[f"task_{task_id}_overall"]
                        if metric_name == "Accuracy"
                        else metric_value_dict[f"task_{task_id}_eval_loss"]
                    )
                    metric_name = f"{metric_name}/Evaluation" if metric_name == "Loss" else metric_name
                    self.log(metric_name=metric_name, metric_value=metric_value_dict)
                end_time_eval = time.time()
                print(f"Evaluation took: {end_time_eval - start_time_eval}s.")

            if 0 < self.config.max_steps == self.global_step:
                break

    def train_iter(self, training_instance: Tuple[Tensor, Tensor], task_id: str, encoder_only: bool = False) -> float:
        # for each iteration of training call this function
        # get the inputs; data is a list of [inputs, labels]
        self.model.train()
        training_features, training_targets = training_instance
        if encoder_only:
            training_features = torch.cat([training_features[0], training_features[1]], dim=0)
        training_features = training_features.to(self.device)
        training_targets = training_targets.to(self.device)

        # zero the parameter gradients
        self.config.optimizer.zero_grad()
        bsz = training_targets.shape[0]

        # forward + backward + optimize
        if encoder_only:
            outputs = self.model(training_features, task_id, encoder_only=encoder_only)
            f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
            outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        else:
            outputs = self.model(training_features, task_id)

        # CE loss
        if encoder_only:
            sup_con_loss_func = SupConLoss(10)
            loss = sup_con_loss_func(features=outputs, labels=training_targets)
        else:
            loss = self.config.criterion(outputs, training_targets)

        if self.config.strategy == "LwF":
            loss = loss + self.cl_strategy.lwf_loss(
                features=training_features, current_model=self.model, current_task_id=task_id
            )
        if self.config.strategy.startswith("EWC"):
            loss += self.cl_strategy.ewc_loss(model=self.model, current_task_id=task_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.config.optimizer.step()
        if self.config.use_scheduler:
            self.scheduler.step()
        self.global_step += 1

        return loss.item()

    def terminate_logging(self):
        for logger in self.config.loggers:
            logger.terminate()

    def log(self, metric_name: str, metric_value: Any, global_step: Optional[int] = None):
        if global_step is None:
            global_step = self.global_step
        for logger in self.config.loggers:
            logger.log_metric(metric_name=metric_name, metric_value=metric_value, global_step=global_step)

    def init_loggers(self):
        # this is for the case of linear probes since they have intended block and we want to avoid over writing
        experiment_name = f"/{self.config.experiment_name}" if self.config.experiment_name is not None else ""
        if hasattr(self.model, "intended_block"):
            tb_logger = TensorboardLogger(
                log_dir="../tb_logs" + experiment_name,
                model_name=f"{self.config.strategy}_{type(self.data_stream).__name__}_{type(self.model).__name__}"
                f"_{self.config.probe_caller}/{self.model.intended_block}",
                seed_value=self.config.seed_value,
            )
            io_logger = IOLogger(
                log_dir="",
                model_name=f"{self.config.strategy}_{type(self.data_stream).__name__}_{type(self.model).__name__}"
                f"_{self.config.probe_caller}/{self.model.intended_block}",
                seed_value=self.config.seed_value,
            )
        else:
            tb_logger = TensorboardLogger(
                log_dir="../tb_logs" + experiment_name,
                model_name=f"{self.config.strategy}_{type(self.data_stream).__name__}_{type(self.model).__name__}",
                seed_value=self.config.seed_value,
            )
            io_logger = IOLogger(
                log_dir="",
                model_name=f"{self.config.strategy}_{type(self.data_stream).__name__}_{type(self.model).__name__}",
                seed_value=self.config.seed_value,
            )

        self.config.loggers = [io_logger, tb_logger]

    def save_model(self, task_id: str):
        if self.config.verbose:
            print(f"   Training on task {task_id} is over.\n      ^^^Saving model.^^^")
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.config.saving_dir,
                f"{self.config.strategy}_{type(self.data_stream).__name__}_"
                f"{type(self.model).__name__}_{task_id}_{self.config.seed_value}.pt",
            ),
        )


class OneShotTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        config: OneShotConfig = OneShotConfig(),
    ):
        self.valid_dataset = valid_dataset
        self.train_dataset = train_dataset
        self.config = config

        self.hparam_training = asdict(self.config)
        keys_to_remove = ["prediction_evaluator", "criterion", "optimizer", "loggers", "early_stopping_config"]
        for key in keys_to_remove:
            self.hparam_training.pop(key)

        print(f" --> Training hparams:\n --> {self.hparam_training}")

        if self.config.save_progress:
            if not os.path.exists(self.config.saving_dir):
                os.makedirs(self.config.saving_dir)

        if self.config.early_stopping_config is not None:
            self.config.early_stopping_config.model_name = type(model).__name__

        self.n_gpu, self.device = gpu_information_summary(show=self.config.verbose)
        self.model = model
        self.model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.config.optimizer is None:
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
            self.config.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=5e-4,
                amsgrad=True,
            )
        self.global_step = 0
        if self.config.loggers is None:
            self.init_loggers()
        self.scheduler = None
        self.cl_strategy = None
        self.checkpoint = CheckPointManager(
            config=CheckPointingConfig(
                model_name=type(self.model).__name__,
                history=self.config.progress_history,
                verbose=True,
                directory=self.config.saving_dir,
            )
        )
        self.hparam_metrics = None
        self.running_loss = 0.0

    def train(self):
        # for each task call train_tasks
        encoder_train_transform = TwoCropTransform(transform=self.train_dataset.transform)
        encoder_valid_transform = TwoCropTransform(transform=self.valid_dataset.transform)
        if self.config.early_stopping_config is not None:
            if self.config.use_sup_con:
                self.train_dataset.transform = encoder_train_transform
                self.valid_dataset.transform = encoder_valid_transform
                self.train_task(task_training=self.train_dataset, task_validation=self.valid_dataset, encoder_only=True)
                self.train_dataset.transform = encoder_train_transform.transform
                self.valid_dataset.transform = encoder_valid_transform.transform
            else:
                self.train_task(task_training=self.train_dataset, task_validation=self.valid_dataset)
        else:
            if self.config.use_sup_con:
                self.train_dataset.transform = encoder_train_transform
                self.train_task(task_training=self.train_dataset, encoder_only=True)
                self.train_dataset.transform = encoder_train_transform.transform
            else:
                self.train_task(task_training=self.train_dataset)

        if self.config.use_sup_con:
            self.train_task(task_training=self.train_dataset)

        if self.hparam_metrics is not None:
            hparam_logging = (self.hparam_training, self.hparam_metrics)
            self.log(metric_name="hparams", metric_value=hparam_logging)

        self.terminate_logging()

    def train_task(
        self, task_training: Dataset, task_validation: Optional[Dataset] = None, encoder_only: bool = False
    ) -> None:
        # for each task call train_epochs
        train_loader = torch.utils.data.DataLoader(
            task_training,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        if self.config.max_steps > 0:
            self.nb_training_steps = self.config.max_steps
        else:
            self.nb_training_steps = (
                len(train_loader) * self.config.nb_epochs_supcon
                if not encoder_only and self.config.use_sup_con
                else len(train_loader) * self.config.nb_epochs
            )

        if self.config.use_scheduler:
            self.scheduler = cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.config.optimizer,
                num_warmup_steps=self.config.nb_warmup_steps,
                num_training_steps=self.nb_training_steps - self.config.nb_warmup_steps,
                num_cycles=1,
            )

        self.running_loss = 0.0
        if self.config.early_stopping_config is not None:
            early_stopping = EarlyStopping(config=self.config.early_stopping_config)
            validation_monitoring = PredictionBasedEvaluator(
                metrics=[Loss()], batch_size=self.config.batch_size, num_workers=self.config.num_workers
            )
            epoch = 0
            while not early_stopping.early_stop:
                self.train_epoch(train_loader=train_loader, epoch=epoch)
                validation_loss = validation_monitoring.eval_one_task(model=self.model, task=task_validation)["Loss"][
                    "eval_loss"
                ]
                self.log(metric_name="Loss/Evaluation", metric_value=validation_loss)
                early_stopping(val_loss=validation_loss, model=self.model)
                epoch += 1

            # load the last checkpoint with the best model
            self.model.load_state_dict(torch.load(self.config.early_stopping_config.path))

        else:
            epochs = (
                self.config.nb_epochs_supcon if self.config.use_sup_con and not encoder_only else self.config.nb_epochs
            )
            for epoch in tqdm(range(epochs), desc="Epoch progress"):
                self.train_epoch(train_loader=train_loader, epoch=epoch, encoder_only=encoder_only)
                if self.config.save_progress:
                    self.checkpoint(model=self.model, step=epoch, optimizer=self.config.optimizer)
                if 0 < self.config.max_steps == self.global_step:
                    break

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, encoder_only: bool = False) -> None:
        # for each epoch call train_iter
        for batch_number, training_instance in enumerate(train_loader):
            self.running_loss += self.train_iter(training_instance=training_instance, encoder_only=encoder_only)

            if self.global_step % 10 == 9:
                print("[Epoch %d, Batch %5d] loss: %.3f" % (epoch + 1, batch_number + 1, self.running_loss / 10))
                self.log(metric_name="Loss/Training", metric_value=self.running_loss / 10)
                self.log(metric_name="LR", metric_value=self.config.optimizer.param_groups[0]["lr"])
                self.running_loss = 0.0

            if (
                (
                    self.config.logging_step > 0
                    and self.global_step % self.config.logging_step == (self.config.logging_step - 1)
                )
                and self.config.prediction_evaluator is not None
                and not encoder_only
            ):
                start_time_eval = time.time()
                evaluator_results = self.config.prediction_evaluator.eval_one_task(
                    model=self.model, task=self.valid_dataset, nb_classes=self.config.nb_classes
                )
                self.hparam_metrics = dict()
                for metric_name, metric_value in evaluator_results.items():
                    self.hparam_metrics[f"hparams/{metric_name}"] = (
                        metric_value["overall"] if metric_name == "Accuracy" else metric_value["eval_loss"]
                    )
                    metric_name = f"{metric_name}/Evaluation" if metric_name == "Loss" else metric_name

                    self.log(metric_name=metric_name, metric_value=metric_value)
                end_time_eval = time.time()
                print(f"Evaluation took: {end_time_eval - start_time_eval}s.")

            if 0 < self.config.max_steps == self.global_step:
                break

    def train_iter(self, training_instance: Tuple[Tensor, Tensor], encoder_only: bool = False) -> float:
        # for each iteration of training call this function
        # get the inputs; data is a list of [inputs, labels]
        self.model.train()
        training_features, training_targets = training_instance
        if encoder_only:
            training_features = torch.cat([training_features[0], training_features[1]], dim=0)
        training_features = training_features.to(self.device)
        training_targets = training_targets.to(self.device)

        # zero the parameter gradients
        self.config.optimizer.zero_grad()
        bsz = training_targets.shape[0]

        # forward + backward + optimize
        if encoder_only:
            outputs = self.model(training_features, encoder_only=encoder_only)
            f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
            outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        else:
            outputs = self.model(training_features)
        # CE loss
        if encoder_only:
            sup_con_loss_func = SupConLoss(10)
            loss = sup_con_loss_func(features=outputs, labels=training_targets)
        else:
            loss = self.config.criterion(outputs, training_targets)

        if self.n_gpu > 1:
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.config.optimizer.step()
        if self.config.use_scheduler:
            self.scheduler.step()
        self.global_step += 1

        return loss.item()

    def terminate_logging(self):
        for logger in self.config.loggers:
            logger.terminate()

    def log(self, metric_name: str, metric_value: Any, global_step: Optional[int] = None):
        if global_step is None:
            global_step = self.global_step
        for logger in self.config.loggers:
            logger.log_metric(metric_name=metric_name, metric_value=metric_value, global_step=global_step)

    def init_loggers(self):
        # this is for the case of linear probes since they have intended block and we want to avoid over writing
        experiment_name = f"/{self.config.experiment_name}" if self.config.experiment_name is not None else ""
        tb_logger = TensorboardLogger(
            log_dir="../tb_logs" + experiment_name,
            model_name=f"{type(self.model).__name__}",
            seed_value=self.config.seed_value,
        )
        io_logger = IOLogger(
            log_dir="",
            model_name=f"{type(self.model).__name__}",
            seed_value=self.config.seed_value,
        )

        self.config.loggers = [io_logger, tb_logger]

    @property
    def train_batch_size(self):
        return self.config.batch_size * max(1, self.n_gpu)
