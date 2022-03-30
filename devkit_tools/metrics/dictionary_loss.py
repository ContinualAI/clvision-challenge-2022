################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 18-03-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List, Dict

import torch
from torch import Tensor

from avalanche.evaluation import PluginMetric, Metric, GenericPluginMetric
from avalanche.evaluation.metric_definitions import TResult
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import Loss
from avalanche.evaluation.metric_utils import phase_and_task, get_metric_name
from collections import defaultdict


class DictionaryLoss(Metric[Dict[str, float]]):

    def __init__(self):
        self._mean_losses = defaultdict(Loss)

    @torch.no_grad()
    def update(
            self,
            loss_dict: Dict[str, Tensor],
            patterns: int,
            task_label: int) -> None:
        """
        Update the running loss given the loss dictionary and the
        minibatch size.

        :param loss: The loss dictionary. Different reduction types don't
            affect the result.
        :param patterns: The number of patterns in the minibatch.
        :param task_label: the task label associated to the current experience
        :return: None.
        """
        for loss_name, loss_value in loss_dict.items():
            self._mean_losses[loss_name].update(
                torch.mean(loss_value), patterns=patterns,
                task_label=task_label)

    def result(self, task_label=None) -> Dict[int, Dict[str, float]]:
        """
        Retrieves the running average loss per pattern.

        Calling this method will not change the internal state of the metric.
        :param task_label: None to return metric values for all the task labels.
            If an int, return value only for that task label
        :return: The running loss, as a dictionary of
        task_label -> dict[loss_name -> loss_mean].
        """
        assert task_label is None or isinstance(task_label, int)
        result_dict: Dict[int, Dict[str, float]] = defaultdict(dict)
        if task_label is None:
            for loss_name, loss_metric in self._mean_losses.items():
                l_res = loss_metric.result()
                for task_label, loss in l_res.items():
                    result_dict[task_label][loss_name] = loss
        else:
            for loss_name, loss_metric in self._mean_losses.items():
                l_res = loss_metric.result(task_label=task_label)
                result_dict[task_label][loss_name] = l_res[task_label]
        return result_dict

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.

        :param task_label: None to reset all metric values. If an int,
            reset metric value corresponding to that task label.
        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_losses = defaultdict(Loss)
        else:
            for loss_metric in self._mean_losses.values():
                loss_metric.reset(task_label=task_label)


class ExtendedGenericPluginMetric(GenericPluginMetric[TResult]):
    """
    A generified version of :class:`GenericPluginMetric` which supports emitting
    multiple metric values from a single metric instance.

    Child classes need to emit metric values via `result()` as dictionaries
    [metric_name -> value]. The resulting metric name will be:
    `str(self)_metric_name`.
    """

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result(strategy)
        add_exp = self._emit_at == "experience"
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v_d in metric_value.items():
                if not isinstance(v_d, dict):
                    v_d = {str(self): v_d}

                for n, v in v_d.items():
                    metric_name = get_metric_name(
                        str(self) + '_' + n,
                        strategy,
                        add_experience=add_exp,
                        add_task=k
                    )
                    metrics.append(
                        MetricValue(self, metric_name, v, plot_x_position)
                    )
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [
                MetricValue(self, metric_name, metric_value, plot_x_position)
            ]


class DictLossPluginMetric(ExtendedGenericPluginMetric[Dict[str, float]]):
    """
    Base class for metrics managing dictionary of losses.

    Dictionaries of losses are commonly used in detection and segmentation
    tasks, where each loss_name -> loss_tensor pair represents the overall
    loss contribution from each component.

    This metric works on the 'train' phase only.
    """
    def __init__(self, reset_at, emit_at, *, dictionary_name='loss_dict'):
        self._dict_loss = DictionaryLoss()
        super(DictLossPluginMetric, self).__init__(
            self._dict_loss, reset_at, emit_at, mode='train'
        )
        self.dictionary_name = dictionary_name

    def reset(self, strategy=None) -> None:
        if self._reset_at == "stream" or strategy is None:
            self._dict_loss.reset()
        else:
            self._dict_loss.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None):
        if self._emit_at == "stream" or strategy is None:
            return self._dict_loss.result()
        else:
            return self._dict_loss.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]
        self._dict_loss.update(
            getattr(strategy, self.dictionary_name),
            patterns=len(strategy.mb_y),
            task_label=task_label
        )


class DictMinibatchLoss(DictLossPluginMetric):
    """
    The minibatch dictionary loss metric.
    This plugin metric only works at training time.

    This metric computes the average loss over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`DictEpochLoss` instead.
    """

    def __init__(self, dictionary_name='loss_dict'):
        """
        Creates an instance of the DictMinibatchLoss metric.
        """
        super().__init__(
            reset_at="iteration", emit_at="iteration",
            dictionary_name=dictionary_name
        )

    def __str__(self):
        return "DictLoss_MB"


class DictEpochLoss(DictLossPluginMetric):
    """
    The average loss over a single training epoch.
    This plugin metric only works at training time.

    The loss will be logged after each training epoch by computing
    the loss on the predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, dictionary_name='loss_dict'):
        """
        Creates an instance of the DictEpochLoss metric.
        """

        super().__init__(
            reset_at="epoch", emit_at="epoch",
            dictionary_name=dictionary_name
        )

    def __str__(self):
        return "DictLoss_Epoch"


class DictRunningEpochLoss(DictLossPluginMetric):
    """
    The average loss across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the loss averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, dictionary_name='loss_dict'):
        """
        Creates an instance of the RunningEpochLoss metric.
        """

        super().__init__(
            reset_at="epoch", emit_at="iteration",
            dictionary_name=dictionary_name
        )

    def __str__(self):
        return "DictRunningLoss_Epoch"


def dict_loss_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    dictionary_name='loss_dict'
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the dictionary of minibatch losses at training time.
    :param epoch: If True, will return a metric able to log
        the dictionary of epoch losses at training time.
    :param epoch_running: If True, will return a metric able to log
        the dictionary of running epoch losses at training time.
    :param dictionary_name: The name of the dictionary to monitor.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(DictMinibatchLoss(dictionary_name=dictionary_name))

    if epoch:
        metrics.append(DictEpochLoss(dictionary_name=dictionary_name))

    if epoch_running:
        metrics.append(DictRunningEpochLoss(dictionary_name=dictionary_name))

    return metrics


__all__ = [
    "DictionaryLoss",
    "DictMinibatchLoss",
    "DictEpochLoss",
    "DictRunningEpochLoss",
    "dict_loss_metrics"
]
