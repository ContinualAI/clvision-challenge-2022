from pathlib import Path
from typing import Union

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from devkit_tools import ChallengeClassificationDataset
from devkit_tools.challenge_constants import \
    DEMO_CLASSIFICATION_FORCED_TRANSFORMS, DEMO_CLASSIFICATION_EXPERIENCES
from ego_objects import EgoObjects


def demo_classification_benchmark(
        dataset_path: Union[str, Path],
        class_order_seed: int,
        *,
        train_transform=None,
        eval_transform=None,
        train_json_name=None,
        test_json_name=None,
        instance_level=True,
        n_exps=DEMO_CLASSIFICATION_EXPERIENCES):

    base_transforms = dict(
        train=(DEMO_CLASSIFICATION_FORCED_TRANSFORMS, None),
        eval=(DEMO_CLASSIFICATION_FORCED_TRANSFORMS, None)
    )

    ego_api = None
    if train_json_name is not None:
        ego_api = EgoObjects(str(Path(dataset_path) / train_json_name))

    train_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=ego_api,
        train=True,
        bbox_margin=20,
        instance_level=instance_level
    )

    ego_api = None
    if test_json_name is not None:
        ego_api = EgoObjects(str(Path(dataset_path) / test_json_name))

    test_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=ego_api,
        train=False,
        bbox_margin=20,
        instance_level=instance_level
    )

    avl_train_dataset = AvalancheDataset(
        train_dataset,
        transform_groups=base_transforms,
        initial_transform_group='train'
    ).freeze_transforms()

    avl_test_dataset = AvalancheDataset(
        test_dataset,
        transform_groups=base_transforms,
        initial_transform_group='eval'
    ).freeze_transforms()

    return nc_benchmark(
        avl_train_dataset,
        avl_test_dataset,
        n_experiences=n_exps,
        task_labels=False,
        seed=class_order_seed,
        shuffle=True,
        train_transform=train_transform,
        eval_transform=eval_transform
    )


__all__ = [
    'demo_classification_benchmark'
]
