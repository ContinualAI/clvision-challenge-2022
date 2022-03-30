from typing import Union, List, Callable, Any, Sequence

from avalanche.evaluation.metrics.detection import DetectionEvaluator, SupportedDatasetApiDef, \
    DetectionMetrics
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
    return DetectionMetrics(
        evaluator_factory=evaluator_factory,
        gt_api_def=gt_api_def,
        save_folder=save_folder,
        filename_prefix=filename_prefix,
        iou_types=iou_types,
        summarize_to_stdout=summarize_to_stdout)


__all__ = [
    'make_ego_objects_metrics'
]
