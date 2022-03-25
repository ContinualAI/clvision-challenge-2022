import json
import os

import torchvision
from lvis import LVIS
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection

from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset, \
    AvalancheDataset
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metrics.detection import TensorEncoder
from devkit_tools.metrics.ego_evaluator import EgoEvaluator
from ego_objects import EgoObjects


class EgoMetrics(PluginMetric[str]):
    """
    This metric serializes model outputs to JSON files.
    The metric produces one file for each evaluation experience.
    It also returns the metrics computed by Ego benchmark based
    on model output. Metrics are returned after each
    evaluation experience.
    """

    def __init__(self, save_folder=None, filename_prefix='model_output',
                 stream='test', iou_types=['bbox']):
        """
        :param save_folder: path to the folder where to write model output
            files. None to disable writing to file.
        :param filename_prefix: prefix common to all model outputs files
        :param iou_types: list of iou types. Defaults to ['bbox'].
        """
        super().__init__()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

        self.save_folder = save_folder
        self.filename_prefix = filename_prefix
        self.iou_types = iou_types
        self.stream = stream

        self.last_train_exp = -1
        """
        The last train experience.
        """

        self.ego_evaluator = None
        """Main Ego evaluator object to compute metrics"""

        self.current_filename = None
        """File containing current model dump"""

        self.current_outputs = []
        """List of dictionaries containing the current model outputs"""

        self.no_save = save_folder is None
        """If True, no JSON file will be written"""

    def reset(self) -> None:
        self.current_outputs = []
        self.current_filename = None

    def update(self, res):
        if not self.no_save:
            self.current_outputs.append(res)
        self.lvis_evaluator.update(res)

    def result(self):
        if not self.no_save:
            with open(self.current_filename, 'w') as f:
                json.dump(self.current_outputs, f, cls=TensorEncoder)

        self.lvis_evaluator.evaluate()
        self.lvis_evaluator.summarize()
        # Encode metrics in CodaLab output format
        bbox_eval = self.ego_evaluator.ego_eval_per_iou['bbox']
        score_str = ''
        ordered_keys = sorted(bbox_eval.results.keys())
        for key in ordered_keys:
            value = bbox_eval.results[key]
            score_str += '{}: {:.5f}\n'.format(key, value)
        score_str = score_str[:-1]  # Remove final \n
        print("******* ", score_str)
        return score_str

    def before_training_exp(
            self,
            strategy):
        super().before_training_exp(strategy)

        self.last_train_exp = strategy.experience.current_experience

    def before_eval_exp(self, strategy) -> None:
        super().before_eval_exp(strategy)
        if strategy.experience.origin_stream.name != self.stream:
            return

        self.reset()
        ego_api = get_detection_api_from_dataset(
            strategy.experience.dataset)
        self.lvis_evaluator = EgoEvaluator(ego_api, self.iou_types)
        self.current_filename = self._get_filename(strategy)

    def after_eval_iteration(self, strategy) -> None:
        super().after_eval_iteration(strategy)
        if strategy.experience.origin_stream.name != self.stream:
            return

        self.update(strategy.res)

    def after_eval_exp(self, strategy):
        super().after_eval_exp(strategy)
        if strategy.experience.origin_stream.name != self.stream:
            return

        return self._package_result(strategy)

    def _package_result(self, strategy):
        metric_name = get_metric_name(self, strategy, add_experience=True,
                                      add_task=False)
        plot_x_position = strategy.clock.train_iterations
        filename = self.result()
        metric_values = [
            MetricValue(self, metric_name, filename, plot_x_position)
        ]
        return metric_values

    def _get_filename(self, strategy):
        """e.g. prefix_eval_exp0.json"""
        middle = '_eval_exp'
        if self.filename_prefix == '':
            middle = middle[1:]
        return os.path.join(
            self.save_folder,
            f"{self.filename_prefix}{middle}"
            f"{self.last_train_exp}.json")

    def __str__(self):
        return "EgoMetrics"


def get_detection_api_from_dataset(dataset):
    # Lorenzo: adapted to support LVIS and AvalancheDataset
    for _ in range(100):
        if isinstance(dataset, CocoDetection):
            break
        elif isinstance(dataset, (LVIS, EgoObjects)):
            break
        elif isinstance(dataset, EgoObjects):
            break
        elif hasattr(dataset, 'lvis_api'):
            break
        elif hasattr(dataset, 'ego_api'):
            break
        elif isinstance(dataset, Subset):
            dataset = dataset.dataset
        elif isinstance(dataset, AvalancheSubset):
            dataset = dataset._original_dataset
        elif isinstance(dataset, AvalancheConcatDataset):
            dataset = dataset._dataset_list[0]
        elif isinstance(dataset, AvalancheDataset):
            dataset = dataset._dataset

    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if isinstance(dataset, (LVIS, EgoObjects)):
        return dataset
    if hasattr(dataset, 'lvis_api'):
        return dataset.lvis_api
    if hasattr(dataset, 'ego_api'):
        return dataset.ego_api

    raise RuntimeError('Could not find the API object')
