import json
import warnings
from typing import Union, Dict, Any, Set
from pathlib import Path

import torch
from torch import Tensor

from avalanche.benchmarks import GenericCLScenario
from avalanche.core import SupervisedPlugin, Template
from avalanche.training.templates import SupervisedTemplate


class ClassificationOutputExporter(SupervisedPlugin[SupervisedTemplate]):
    """
    This plugin serializes model outputs for classification tasks.
    The plugin produces one file for each evaluation experience.

    Model outputs will be used to compute the final score.
    """

    def __init__(
            self,
            benchmark: GenericCLScenario,
            save_folder: Union[str, Path] = None,
            filename_prefix='track1_output',
            stream='test',
            strict=False):
        """
        Creates an instance of the output exporter plugin.

        :param benchmark: the benchmark to use.
        :param save_folder: path to the folder where to write model output
            files. None to disable writing to file.
        :param filename_prefix: prefix common to all model outputs files
        :param stream: the name of the stream.
        :param strict: if True, it will raise an error if the correct order
            of train/eval phases is not followed.
        """
        super().__init__()

        if save_folder is not None:
            save_folder = Path(save_folder)
        else:
            save_folder = Path.cwd()

        save_folder.mkdir(exist_ok=True, parents=True)

        self.n_test_experiences = len(benchmark.streams[stream])
        """
        How many test experiences are there in the benchmark.
        """

        self.strict = strict
        """
        If True, will raise and error instead of reporting warnings.
        """

        self.save_folder = save_folder
        """
        The folder in which the model outputs will be saved.
        """

        self.filename_prefix = filename_prefix
        """
        The base file name to use to store model outputs.
        """

        self.stream = stream
        """
        The name of the stream for which the model outputs have to be saved.
        """

        self.last_train_exp = -1
        """
        The last train experience.
        """

        self.last_eval_exp = -1
        """
        The last eval experience. Reset at the beginning of each eval phase.
        """

        self._last_eval_completed = True
        """
        Used to keep track if the last evaluation completed successfully.
        """

        self._enable_recording = False
        """
        If True, model outputs will be recorded and saved.
        To prevent considering outputs resulting from intermediate validations.
        """

        self.current_outputs: Dict[int, Any] = dict()
        """
        Dictionary mapping the test experience to its model outputs.
        """

    def reset(self) -> None:
        self.current_outputs = dict()
        self.last_eval_exp = -1

    def update(self, res):
        if not isinstance(res, Tensor):
            res = torch.as_tensor(res)

        res = res.detach().cpu()

        res = self._convert_predictions(res)
        self.current_outputs[self.last_eval_exp].append(res)

    def before_training(self, strategy: Template, *args, **kwargs):
        self.reset()
        self._enable_recording = False  # Prevent saving validation outputs

    def after_training(self, strategy: Template, *args, **kwargs):
        self._enable_recording = True

    def before_training_exp(
            self,
            strategy: SupervisedTemplate,
            *args, **kwargs):
        super().before_training_exp(strategy, *args, **kwargs)
        if not self._last_eval_completed:
            err_msg = \
                'In the last eval phase you haven\'t tested on (at least) ' \
                'the growing test set: the produced model outputs may not ' \
                'make sense and they can\'t be used as a valid solution.'
            if self.strict:
                raise RuntimeError(err_msg)

            warnings.warn(err_msg)

        self._last_eval_completed = False

        new_exp_id = strategy.experience.current_experience
        if (self.last_train_exp + 1) != new_exp_id:
            err_msg = \
                'You are not following the correct train/eval loop: ' \
                'the produced model outputs may not make sense and they ' \
                'can\'t be used as a valid solution.'
            if self.strict:
                raise RuntimeError(err_msg)

            warnings.warn(err_msg)

        self.last_train_exp = new_exp_id

    def before_eval_exp(
            self, strategy: SupervisedTemplate, *args, **kwargs):
        super().before_eval_exp(strategy, *args, **kwargs)
        if strategy.experience.origin_stream.name != self.stream:
            return

        if not self._enable_recording:
            return

        self.last_eval_exp = strategy.experience.current_experience
        self.current_outputs[self.last_eval_exp] = []

    def after_eval_iteration(
            self, strategy: SupervisedTemplate, *args, **kwargs):
        super().after_eval_iteration(strategy, *args, **kwargs)
        if strategy.experience.origin_stream.name != self.stream:
            return

        if not self._enable_recording:
            return

        assert self.last_eval_exp >= 0
        self.update(strategy.mb_output)

    def after_eval_exp(self, strategy, *args, **kwargs):
        super().after_eval_exp(strategy)
        if strategy.experience.origin_stream.name != self.stream:
            return

        if not self._enable_recording:
            return

        if self._last_eval_completed:
            return

        if self.last_train_exp < 0:
            # Eval executed before a train loop
            return

        evaluated_exps = set(self.current_outputs.keys())
        test_set_exps = self._needed_test_experiences()

        evaluated_exps = evaluated_exps.intersection(test_set_exps)
        if len(evaluated_exps) != len(test_set_exps):
            # Not all epxs from the growing test have been evaluated yet
            return

        self._last_eval_completed = True
        self._save_results()

    def _save_results(self):
        test_set_exps = self._needed_test_experiences()
        test_set_exps = sorted(list(test_set_exps))

        out_tensors = dict()

        for exp_id in test_set_exps:
            exp_result = self.current_outputs[exp_id]
            exp_out_tensor = torch.cat(exp_result)
            out_tensors[exp_id] = exp_out_tensor.tolist()

        result_file_path = self._get_filename()
        with open(result_file_path, 'w') as f:
            json.dump(out_tensors, f)

    def _needed_test_experiences(self) -> Set[int]:
        return set(range(self.n_test_experiences))

    def _get_filename(self):
        """e.g. prefix_eval_exp0.json"""
        middle = '_eval_exp'
        if self.filename_prefix == '':
            middle = middle[1:]
        f_name = f"{self.filename_prefix}{middle}{self.last_train_exp}.json"
        return self.save_folder / f_name

    @staticmethod
    def _convert_predictions(predicted_y: Tensor) -> Tensor:
        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        predicted_y = predicted_y.to(torch.int64)
        return predicted_y

    def __str__(self):
        return "ClassificationOutputExporter"


__all__ = [
    'ClassificationOutputExporter'
]
