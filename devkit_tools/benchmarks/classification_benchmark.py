import hashlib
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
from typing_extensions import Literal

from avalanche.benchmarks import nc_benchmark, dataset_benchmark, \
    GenericCLScenario
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset
from devkit_tools import ChallengeClassificationDataset
from devkit_tools.challenge_constants import \
    DEMO_CLASSIFICATION_EXPERIENCES, \
    CHALLENGE_CLASSIFICATION_EXPERIENCES, \
    CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS, \
    DEFAULT_CHALLENGE_CLASS_ORDER_SEED, DEFAULT_DEMO_TRAIN_JSON, \
    DEFAULT_DEMO_TEST_JSON, DEFAULT_CHALLENGE_TRAIN_JSON, \
    DEFAULT_CHALLENGE_TEST_JSON
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
        n_validation_videos=0,
        validation_video_selection_seed=1337,
        unlabeled_test_set=True,
        remainder_classes_allocation: Literal['exp0', 'initial_exps'] =
        'exp0'):
    """
    Creates the challenge instance classification benchmark.

    Please don't change this code. You are free to customize the dataset
    path and jsons paths, the train_transform, and eval_transform parameters.
    Don't change other parameters or the code.

    Images will be loaded as 224x224 by default. You are free to resize them
    by adding an additional transformation atop of the mandatory one.

    Note: calling this method may change the global state of random number
    generators (PyTorch, NumPy, Python's "random").

    For the validation datasets, test transformations are used. In addition,
    the transforms group will be again enforced as "eval" in the strategy (in
    the `eval_dataset_adaptation` method). Setting the initial group to eval
    now allows for a smoother use of these datasets by participants that
    do not plan to use Avalanche strategies. One can use the validation set
    with training transformations by obtaining a new view of the dataset as
    follows: `val_set_with_train_transforms = val_dataset.train()`.

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
    :param n_validation_videos: How many validation videos per class.
        Defaults to 0, which means that no validation stream will be created.
    :param validation_video_selection_seed: The seed to use when selecting
        videos to allocate to the validation stream.
    :param unlabeled_test_set: If True, the test stream will be made of
        a single test set. Defaults to True.
    :param remainder_classes_allocation: How to manage the remainder classes
        (in the case that  overall_classes % n_exps > 0). Default to 'exp0'.
    :return: The instance classification benchmark.
    """

    base_transforms = dict(
        train=(CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS, None),
        eval=(CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS, None)
    )

    if train_json_name is None:
        train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON
    train_ego_api = EgoObjects(str(Path(dataset_path) / train_json_name))

    if test_json_name is None:
        test_json_name = DEFAULT_CHALLENGE_TEST_JSON
    test_ego_api = EgoObjects(str(Path(dataset_path) / test_json_name))

    train_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=train_ego_api,
        train=True,
        bbox_margin=20,
        instance_level=instance_level
    )

    test_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=test_ego_api,
        train=False,
        bbox_margin=20,
        instance_level=instance_level
    )

    # validation_cls_imgs = defaultdict(list)
    val_img_ids = set()
    if n_validation_videos > 0:
        random.seed(validation_video_selection_seed)
        class_to_videos_dict = ChallengeClassificationDataset.class_to_videos(
            ego_api=train_ego_api, img_ids=train_dataset.img_ids,
            instance_level=instance_level
        )

        max_val_videos = min(len(x) for x in class_to_videos_dict.values()) - 1
        if n_validation_videos > max_val_videos:
            raise ValueError(
                f'Invalid n_validation_videos. Insufficient videos '
                f'(maximum: {max_val_videos})')

        # Iterating through sorted(...) IDs ensures that, given a seed,
        # the same videos are selected every run (determinism)
        classes_ids = sorted(class_to_videos_dict.keys())
        for cls_id in classes_ids:
            videos_dict = class_to_videos_dict[cls_id]
            video_ids = list(sorted(videos_dict.keys()))

            selected_val_videos = random.sample(
                video_ids, k=n_validation_videos)

            for video_id in selected_val_videos:
                # validation_cls_imgs[cls_id].extend(videos_dict[video_id])
                val_img_ids.update(videos_dict[video_id])

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

    benchmark = nc_benchmark(
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

    orig_targets_order_hash = _hash_classes_order(benchmark, 'train')

    if n_validation_videos > 0:
        prev_test_assignment = benchmark.test_exps_patterns_assignment
        n_classes = benchmark.n_classes
        train_assignment = []
        valid_assignment = []

        train_order = benchmark.train_exps_patterns_assignment
        for exp_id in range(len(train_order)):
            train_inst_idx_in_exp = []
            val_inst_idx_in_exp = []
            for instance_idx in train_order[exp_id]:
                img_id = train_dataset.img_ids[instance_idx]
                if img_id in val_img_ids:
                    val_inst_idx_in_exp.append(instance_idx)
                else:
                    train_inst_idx_in_exp.append(instance_idx)

            train_assignment.append(train_inst_idx_in_exp)
            valid_assignment.append(val_inst_idx_in_exp)

        valid_datasets = []
        for exp_id in range(len(benchmark.train_stream)):
            valid_datasets.append(
                AvalancheSubset(
                    avl_train_dataset,
                    indices=valid_assignment[exp_id],
                    initial_transform_group='eval'))

        benchmark = dataset_benchmark(
            train_datasets=[
                AvalancheSubset(
                    avl_train_dataset,
                    indices=train_assignment[exp_id])
                for exp_id in range(len(benchmark.train_stream))
            ],
            test_datasets=[avl_test_dataset],
            other_streams_datasets=dict(valid=valid_datasets),
            complete_test_set_only=True,
            train_transform=train_transform,
            eval_transform=eval_transform
        )

        benchmark.test_exps_patterns_assignment = prev_test_assignment
        benchmark.train_exps_patterns_assignment = train_assignment
        benchmark.valid_exps_patterns_assignment = valid_assignment
        benchmark.n_classes = n_classes

    if unlabeled_test_set:
        prev_test_assignment = benchmark.test_exps_patterns_assignment
        prev_train_assignment = benchmark.train_exps_patterns_assignment
        prev_val_assignment = None
        n_classes = benchmark.n_classes

        other_streams_datasets = None

        if 'valid' in benchmark.streams:
            prev_val_assignment = benchmark.valid_exps_patterns_assignment
            valid_datasets = []
            for exp_id in range(len(benchmark.valid_stream)):
                valid_datasets.append(
                    AvalancheSubset(
                        avl_train_dataset,
                        indices=prev_val_assignment[exp_id],
                        initial_transform_group='eval'))
            other_streams_datasets = dict(valid=valid_datasets)

        benchmark = dataset_benchmark(
            train_datasets=[
                AvalancheSubset(
                    avl_train_dataset,
                    indices=prev_train_assignment[exp_id])
                for exp_id in range(len(benchmark.train_stream))
            ],
            test_datasets=[avl_test_dataset],
            other_streams_datasets=other_streams_datasets,
            complete_test_set_only=True,
            train_transform=train_transform,
            eval_transform=eval_transform
        )

        benchmark.test_exps_patterns_assignment = prev_test_assignment
        benchmark.train_exps_patterns_assignment = prev_train_assignment

        if prev_val_assignment is not None:
            benchmark.valid_exps_patterns_assignment = prev_val_assignment

        benchmark.n_classes = n_classes

    # Check the complex benchmark manipulations (like creating the validation
    # set) hasn't changed the order of classes.
    assert _hash_classes_order(benchmark, 'train') == orig_targets_order_hash
    if 'valid' in benchmark.streams:
        assert _hash_classes_order(benchmark, 'valid') == \
               orig_targets_order_hash

    return benchmark


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

    if 'unlabeled_test_set' not in kwargs:
        kwargs['unlabeled_test_set'] = False

    warnings.warn('You are using the demo benchmark. For the competition, '
                  'please use challenge_classification_benchmark instead.')

    return challenge_classification_benchmark(
        dataset_path=dataset_path,
        class_order_seed=class_order_seed,
        **kwargs
    )


def _hash_classes_order(benchmark: GenericCLScenario, stream_name: str):
    hasher = hashlib.md5()
    stream = benchmark.streams[stream_name]
    for exp_id in range(len(stream)):
        hasher.update(np.int64(exp_id))
        hasher.update(np.array(
            list(sorted(set(stream[exp_id].dataset.targets)))))

    return hasher.hexdigest()


__all__ = [
    'challenge_classification_benchmark',
    'demo_classification_benchmark'
]
