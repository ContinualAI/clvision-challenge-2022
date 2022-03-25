import warnings
from pathlib import Path
from typing import Union
from typing_extensions import Literal

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from devkit_tools import ChallengeClassificationDataset
from devkit_tools.challenge_constants import \
    DEMO_CLASSIFICATION_EXPERIENCES, \
    CHALLENGE_CLASSIFICATION_EXPERIENCES, \
    CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS, \
    DEFAULT_CHALLENGE_CLASS_ORDER_SEED, DEFAULT_DEMO_TRAIN_JSON, \
    DEFAULT_DEMO_TEST_JSON
from ego_objects import EgoObjects


def challenge_classification_benchmark(
        dataset_path: Union[str, Path],
        *,
        class_order_seed: int = DEFAULT_CHALLENGE_CLASS_ORDER_SEED,
        train_transform=None,
        eval_transform=None,
        train_json_name=None,
        test_json_name=None,
        instance_level=True,
        n_exps=CHALLENGE_CLASSIFICATION_EXPERIENCES,
        remainder_classes_allocation: Literal['exp0', 'initial_exps'] = 'exp0'):
    """
    Creates the challenge instance classification benchmark.

    Please don't change this code. You are free to customize the dataset
    path and jsons paths, the train_transform, and eval_transform parameters.
    Don't change other parameters or the code.

    Images will be loaded as 224x224 by default. You are free to resize them
    by adding an additional transformation atop of the mandatory one.

    :param dataset_path: The dataset path.
    :param class_order_seed: The seed defining the order of classes.
        Use DEFAULT_CHALLENGE_CLASS_ORDER_SEED to use the reference order.
    :param train_transform: The train transformations.
    :param eval_transform: The test transformations.
    :param train_json_name: The name of the json file containing the training
        set annotations.
    :param test_json_name: The name of the json file containing the test
        set annotations.
    :param instance_level: If True, creates an instance-based classification
        benchmark. Defaults to True.
    :param n_exps: The number of experiences in the training set.
    :param remainder_classes_allocation: How to manage the remainder classes
        (in the case that  overall_classes % n_exps > 0). Default to 'exp0'.
    :return: The classification benchmark.
    """

    base_transforms = dict(
        train=(CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS, None),
        eval=(CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS, None)
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

    unique_classes = set(avl_train_dataset.targets)
    base_n_classes = len(unique_classes) // n_exps
    remainder_n_classes = len(unique_classes) % n_exps

    per_exp_classes = None
    if remainder_n_classes > 0:
        if remainder_classes_allocation == 'exp0':
            per_exp_classes = {0: base_n_classes + remainder_n_classes}
        elif remainder_classes_allocation == 'initial_exps':
            per_exp_classes = dict()
            for exp_id in range(remainder_n_classes):
                per_exp_classes[exp_id] = base_n_classes + 1

    # print('per_exp_classes', per_exp_classes)
    # print('base_n_classes', base_n_classes)

    return nc_benchmark(
        avl_train_dataset,
        avl_test_dataset,
        n_experiences=n_exps,
        task_labels=False,
        seed=class_order_seed,
        shuffle=True,
        per_exp_classes=per_exp_classes,
        train_transform=train_transform,
        eval_transform=eval_transform
    )


def demo_classification_benchmark(
        dataset_path: Union[str, Path],
        class_order_seed: int,
        **kwargs):
    if 'n_exps' not in kwargs:
        kwargs['n_exps'] = DEMO_CLASSIFICATION_EXPERIENCES

    if 'train_json_name' not in kwargs:
        kwargs['train_json_name'] = DEFAULT_DEMO_TRAIN_JSON

    if 'test_json_name' not in kwargs:
        kwargs['test_json_name'] = DEFAULT_DEMO_TEST_JSON

    warnings.warn('You are using the demo benchmark. For the competition, '
                  'please use challenge_classification_benchmark instead.')

    return challenge_classification_benchmark(
        dataset_path=dataset_path,
        class_order_seed=class_order_seed,
        **kwargs
    )


__all__ = [
    'challenge_classification_benchmark',
    'demo_classification_benchmark'
]
