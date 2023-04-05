import torch.nn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from utilities.utils import zero_like_params_dict, copy_params_dict


class EWCStrategy:
    """
    EWC computes importance of each weight at the end of training on current
    task. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on the previous task in proportion to their importance
    on that task. The importance factors are computed with an additional pass on the
    training set.
    """

    def __init__(self, scenario: str, ewc_lambda: float = 8_000):
        self.ewc_lambda = ewc_lambda
        self.scenario = scenario.lower().split("2")

        self.saved_parameters = dict()
        self.importance_matrices = dict()

    @staticmethod
    def _compute_importance(
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        dataset: Dataset,
        batch_size: int,
        current_task_id: str,
    ):
        """
        Compute EWC importance matrix for each parameter
        """
        model.eval()
        device = next(model.parameters()).device
        importance_matrix = zero_like_params_dict(model=model)
        if current_task_id == "imagenet":
            dataloader = DataLoader(dataset, batch_size=256, num_workers=12)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12)

        for training_instance in dataloader:
            training_features, training_targets = tuple([tensor.to(device) for tensor in training_instance])
            optimizer.zero_grad()
            predictions = model(features=training_features, task_id=current_task_id)
            loss = criterion(predictions, training_targets)
            loss.backward()
            for (net_param_name, net_param_value), (imp_param_name, imp_param_value) in zip(
                model.named_parameters(), importance_matrix
            ):
                assert net_param_name == imp_param_name
                if net_param_value.grad is not None:
                    imp_param_value += net_param_value.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp_param_value in importance_matrix:
            imp_param_value /= float(len(dataloader))

        return importance_matrix

    def ewc_loss(self, model: Module, current_task_id: str):
        if current_task_id == self.scenario[0]:
            return 0
        loss = 0
        for task_id in self.scenario:
            if task_id == current_task_id:
                break
            for (_, current_parameters), (_, saved_parameters), (_, importance_weight) in zip(
                model.named_parameters(), self.saved_parameters[task_id], self.importance_matrices[task_id]
            ):
                loss += (importance_weight * (current_parameters - saved_parameters).pow(2)).sum()
        return self.ewc_lambda * loss

    def record_state(
        self,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        dataset: Dataset,
        batch_size: int,
        current_task_id: str,
    ):
        # to be called at the end of training each task
        importance_matrix = self._compute_importance(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            batch_size=batch_size,
            current_task_id=current_task_id,
        )

        self.importance_matrices[current_task_id] = importance_matrix
        self.saved_parameters[current_task_id] = copy_params_dict(model)


class EWCStrategy500(EWCStrategy):
    def __init__(self, scenario: str, ewc_lambda: float = 500):
        super(EWCStrategy500, self).__init__(scenario, ewc_lambda)


class EWCStrategy8000(EWCStrategy):
    def __init__(self, scenario: str, ewc_lambda: float = 8_000):
        super(EWCStrategy8000, self).__init__(scenario, ewc_lambda)
