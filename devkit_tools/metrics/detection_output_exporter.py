import os
import warnings
from typing import Union, List, Callable, Any, Sequence

from avalanche.evaluation.metrics.detection import DetectionEvaluator, \
    SupportedDatasetApiDef, DetectionMetrics
from avalanche.training.supervised.naive_object_detection import \
    ObjectDetectionTemplate
from devkit_tools.metrics.ego_evaluator import EgoEvaluator
from ego_objects import EgoObjects


def make_ego_objects_metrics(
        save_folder=None,
        filename_prefix='model_output',
        iou_types: Union[str, List[str]] = 'bbox',
        summarize_to_stdout: bool = True,
        evaluator_factory: Callable[[Any, List[str]], DetectionEvaluator] =
        EgoEvaluator,
        gt_api_def: Sequence[SupportedDatasetApiDef] = (('ego_api', EgoObjects),)):
    """
    Returns an instance of :class:`DetectionMetrics` initialized for the
    EgoObjects dataset.

    :param save_folder: path to the folder where to write model output
        files. Defaults to None, which means that the model output of
        test instances will not be stored.
    :param filename_prefix: prefix common to all model outputs files.
        Ignored if `save_folder` is None. Defaults to "model_output"
    :param iou_types: list of (or a single string) strings describing
        the iou types to use when computing metrics.
        Defaults to "bbox". Valid values are "bbox" and "segm".
    :param summarize_to_stdout: if True, a summary of evaluation metrics
        will be printed to stdout (as a table) using the EgoObjects API.
        Defaults to True.
    :param evaluator_factory: Defaults to :class:`EgoObjectEvaluator`
        constructor.
    :param gt_api_def: Defaults to ego object def.
    :return: A metric plugin that can compute metrics (and export outputs
        on the EgoObjects dataset).
    """
    return TestExpIdDetectionMetrics(
        evaluator_factory=evaluator_factory,
        gt_api_def=gt_api_def,
        save_folder=save_folder,
        filename_prefix=filename_prefix,
        iou_types=iou_types,
        summarize_to_stdout=summarize_to_stdout)


class TestExpIdDetectionMetrics(DetectionMetrics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_train_exp = -1
        """
        The last train experience.
        """

    def before_training_exp(self, strategy: "ObjectDetectionTemplate"):
        new_exp_id = strategy.experience.current_experience
        if (self.last_train_exp + 1) != new_exp_id:
            err_msg = \
                'You are not following the correct train/eval loop: ' \
                'the produced model outputs may not make sense and they ' \
                'can\'t be used as a valid solution.'
            warnings.warn(err_msg)
        self.last_train_exp = new_exp_id

    def _get_filename(self, strategy):
        """e.g. prefix_eval_exp0.json"""
        middle = '_eval_exp'
        if self.filename_prefix == '':
            middle = middle[1:]
        f_name = f"{self.filename_prefix}{middle}{self.last_train_exp}.json"
        return os.path.join(self.save_folder, f_name)

    def __str__(self):
        return "EgoObjectsMetrics"


__all__ = [
    'make_ego_objects_metrics'
]
