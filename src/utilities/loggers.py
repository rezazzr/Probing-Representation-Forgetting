import os
import pprint
from abc import abstractmethod, ABC
from typing import Any
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter
import datetime
from prettytable import PrettyTable


class Loggers(ABC):
    @abstractmethod
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        seed_value: int,
    ):
        self.log_dir = log_dir
        self.model_name = model_name
        self.seed_value = seed_value

    @abstractmethod
    def log_metric(self, metric_name: str, metric_value: Any, global_step: int):
        pass

    @abstractmethod
    def terminate(self):
        pass


class TensorboardLogger(Loggers):
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        seed_value: int,
    ):
        super(TensorboardLogger, self).__init__(log_dir, model_name, seed_value)
        experiment_path = os.path.join(
            log_dir, model_name, f"seed_{seed_value}", str(datetime.datetime.now()).replace(" ", "_")
        )
        self.writer = SummaryWriter(experiment_path)

    def log_metric(self, metric_name: str, metric_value: Any, global_step: int):
        if isinstance(metric_value, dict):
            self.writer.add_scalars(metric_name, metric_value, global_step)
        elif isinstance(metric_value, float):
            self.writer.add_scalar(metric_name, metric_value, global_step)
        elif isinstance(metric_value, tuple):
            self.writer.add_hparams(hparam_dict=metric_value[0], metric_dict=metric_value[1])
        elif isinstance(metric_value, Figure):
            # TODO: need to implement this.
            pass
        else:
            raise TypeError(f"metric_value is of type: {type(metric_value).__name__} which is not supported")

    def terminate(self):
        self.writer.close()


class IOLogger(Loggers):
    def __init__(
        self,
        log_dir: str,
        model_name: str,
        seed_value: int,
    ):
        super(IOLogger, self).__init__(log_dir, model_name, seed_value)

    def log_metric(self, metric_name: str, metric_value: Any, global_step: int):
        print("~" * 35)
        print(f"  Model:{self.model_name}, seed:{self.seed_value}, {metric_name} @ step: {global_step}")
        if isinstance(metric_value, dict):
            pprint.pprint(metric_value, indent=2)
        elif isinstance(metric_value, float):
            print(metric_value)
        elif isinstance(metric_value, tuple):
            table = PrettyTable(["KEY", "VALUE"])
            for key, val in metric_value[1].items():
                table.add_row([key, val])
            for key, val in metric_value[0].items():
                table.add_row([key, val])
            print(table)

        elif isinstance(metric_value, Figure):
            pass
        else:
            raise TypeError(f"metric_value is of type: {type(metric_value).__name__} which is not supported")
        print("~" * 35)

    def terminate(self):
        print("-" * 35)
        print(f"  End of Training.")
        print("-" * 35)
