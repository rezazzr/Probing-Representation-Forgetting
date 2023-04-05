import copy

import torch


class LwFStrategy:
    def __init__(self, scenario: str, lambda_0: float = 1, temp: float = 2):
        """
        :param lambda_0: distillation loss scale
        :param temp: temperature used in distillation loss
        """
        self.scenario = scenario.lower().split("2")
        self.temp = temp
        self.lambda_0 = lambda_0
        self.prev_model = None

    def _distillation_loss(self, current_out: torch.Tensor, prev_out: torch.Tensor):
        log_p = torch.log_softmax(current_out / self.temp, dim=1)
        q = torch.softmax(prev_out / self.temp, dim=1)
        result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        return result

    def lwf_loss(self, features: torch.Tensor, current_model: torch.nn.Module, current_task_id: str):
        if self.prev_model is None:
            return 0.0
        predictions_old_tasks_old_model = dict()
        predictions_old_tasks_new_model = dict()
        for task_id in self.scenario:
            if task_id == current_task_id:
                break
            with torch.inference_mode():
                predictions_old_tasks_old_model[task_id] = self.prev_model(features=features, task_id=task_id)
            predictions_old_tasks_new_model[task_id] = current_model(features=features, task_id=task_id)
        dist_loss = 0
        for task_id in predictions_old_tasks_old_model.keys():
            dist_loss += self._distillation_loss(
                current_out=predictions_old_tasks_new_model[task_id],
                prev_out=predictions_old_tasks_old_model[task_id].clone(),
            )
        return self.lambda_0 * dist_loss

    def record_state(self, current_model: torch.nn.Module):
        # to be called at the end of training each task
        self.prev_model = copy.deepcopy(current_model)
