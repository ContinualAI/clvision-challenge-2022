from typing import TypeVar, List, Callable

from avalanche.benchmarks import GenericScenarioStream, GenericExperience, \
    Experience, TScenario, TScenarioStream, GenericCLScenario, TStreamsUserDict
from avalanche.benchmarks.utils import AvalancheDataset


def det_exp_factory(stream: GenericScenarioStream, exp_id: int):
    return DetectionExperience(stream, exp_id)


TDetectionExperience = TypeVar("TDetectionExperience",
                               bound=GenericExperience)


class DetectionExperience(
    Experience[TScenario, TScenarioStream]
):
    def __init__(
        self: TDetectionExperience,
        origin_stream: TScenarioStream,
        current_experience: int,
    ):
        self.origin_stream: TScenarioStream = origin_stream
        self.benchmark: TScenario = origin_stream.benchmark
        self.current_experience: int = current_experience

        self.dataset: AvalancheDataset = (
            origin_stream.benchmark.stream_definitions[
                origin_stream.name
            ].exps_data[current_experience]
        )

    def _get_stream_def(self):
        return self.benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])

    @property
    def task_label(self) -> int:
        if len(self.task_labels) != 1:
            raise ValueError(
                "The task_label property can only be accessed "
                "when the experience contains a single task label"
            )

        return self.task_labels[0]


class DetectionCLScenario(GenericCLScenario[TDetectionExperience]):
    def __init__(
            self,
            n_classes: int,
            *,
            stream_definitions: TStreamsUserDict,
            complete_test_set_only: bool = False,
            experience_factory: Callable[
                ["GenericScenarioStream", int], TDetectionExperience
            ] = None):
        if experience_factory is None:
            experience_factory = DetectionExperience

        super(DetectionCLScenario, self).__init__(
            stream_definitions=stream_definitions,
            complete_test_set_only=complete_test_set_only,
            experience_factory=experience_factory
        )

        self.n_classes = n_classes


__all__ = [
    'det_exp_factory',
    'DetectionExperience',
    'DetectionCLScenario'
]
