import itertools
import warnings
from pathlib import Path
from typing import Union

from avalanche.benchmarks import NCScenario, StreamUserDef
from avalanche.benchmarks.scenarios.detection_scenario import \
    DetectionCLScenario
from avalanche.benchmarks.utils import AvalancheDataset
from devkit_tools import ChallengeClassificationDataset, \
    ChallengeDetectionDataset
from devkit_tools.benchmarks.classification_benchmark import \
    challenge_classification_benchmark
from devkit_tools.benchmarks.make_instance_based import make_instance_based
from devkit_tools.benchmarks.remap_category_ids import \
    make_compact_category_ids_mapping, remap_category_ids
from devkit_tools.challenge_constants import \
    DEFAULT_DEMO_TEST_JSON, \
    DEFAULT_DEMO_TRAIN_JSON, DEMO_DETECTION_EXPERIENCES, \
    CHALLENGE_DETECTION_EXPERIENCES, CHALLENGE_DETECTION_FORCED_TRANSFORMS, \
    DEFAULT_CHALLENGE_TRAIN_JSON, DEFAULT_CHALLENGE_TEST_JSON, \
    DEFAULT_CHALLENGE_CLASS_ORDER_SEED
from ego_objects import EgoObjects


def challenge_category_detection_benchmark(
        dataset_path: Union[str, Path],
        *,
        class_order_seed: int = DEFAULT_CHALLENGE_CLASS_ORDER_SEED,
        train_transform=None,
        eval_transform=None,
        train_json_name=None,
        test_json_name=None,
        n_exps=CHALLENGE_DETECTION_EXPERIENCES,
        n_validation_videos=0,
        validation_video_selection_seed=1337):
    """
    Creates the challenge category detection benchmark.

    Please don't change this code. You are free to customize the dataset
    path and jsons paths, the train_transform, and eval_transform parameters.
    Don't change other parameters or the code.

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
    :param n_exps: The number of experiences in the training set.
    :param n_validation_videos: How many validation videos per category.
        Defaults to 0, which means that no validation stream will be created.
    :param validation_video_selection_seed: The seed to use when selecting
        videos to allocate to the validation stream.
    :return: The category detection benchmark.
    """

    if train_json_name is None:
        train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON

    if test_json_name is None:
        test_json_name = DEFAULT_CHALLENGE_TEST_JSON

    # Use the classification benchmark creator to generate the correct order
    cls_benchmark: NCScenario = challenge_classification_benchmark(
        dataset_path,
        class_order_seed=class_order_seed,
        train_json_name=train_json_name,
        test_json_name=test_json_name,
        instance_level=False,
        n_exps=n_exps,
        unlabeled_test_set=False,
        n_validation_videos=n_validation_videos,
        validation_video_selection_seed=validation_video_selection_seed
    )

    # Create aligned datasets
    train_ego_api = EgoObjects(str(Path(dataset_path) / train_json_name))
    test_ego_api = EgoObjects(str(Path(dataset_path) / test_json_name))

    train_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=train_ego_api,
        train=True,
        bbox_margin=20,
        instance_level=False
    )

    test_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=test_ego_api,
        train=False,
        bbox_margin=20,
        instance_level=False
    )

    # Keep this order
    train_order = cls_benchmark.train_exps_patterns_assignment
    test_order = list(itertools.chain.from_iterable(
        cls_benchmark.test_exps_patterns_assignment))

    train_img_ids = []
    for exp_id in range(len(train_order)):
        img_id_in_exp = []
        for instance_idx in train_order[exp_id]:
            img_id_in_exp.append(train_dataset.img_ids[instance_idx])
        train_img_ids.append(img_id_in_exp)

    test_img_ids = []
    for instance_idx in test_order:
        test_img_ids.append(test_dataset.img_ids[instance_idx])

    valid_img_ids = None
    if 'valid' in cls_benchmark.streams:
        valid_order = cls_benchmark.valid_exps_patterns_assignment
        valid_img_ids = []
        for exp_id in range(len(valid_order)):
            img_id_in_exp = []
            for instance_idx in valid_order[exp_id]:
                img_id_in_exp.append(train_dataset.img_ids[instance_idx])
            valid_img_ids.append(img_id_in_exp)

    base_transforms = dict(
        train=(CHALLENGE_DETECTION_FORCED_TRANSFORMS, None),
        eval=(CHALLENGE_DETECTION_FORCED_TRANSFORMS, None)
    )

    # Align categories IDs
    # The JSON may contain categories with sparse IDs
    # In this way max(categories_ids) >= len(categories), which is not ok!
    # For instance, if category IDs are [0, 1, 2, 3, 5], then initializing
    # the ROI head with n_categories=5 is wrong and it will trigger errors
    # when computing losses (as logits must have 6 elements, not 5)
    # To prevent issues, we just remap categories IDs to range [0, n_categories)
    categories_id_mapping = make_compact_category_ids_mapping(
        train_ego_api, test_ego_api)

    remap_category_ids(train_ego_api, categories_id_mapping)
    remap_category_ids(test_ego_api, categories_id_mapping)

    # Create the training datasets
    train_exps = []
    for exp_id, exp_img_ids in enumerate(train_img_ids):
        exp_dataset = ChallengeDetectionDataset(
            dataset_path,
            train=True,
            ego_api=train_ego_api,
            img_ids=exp_img_ids,
        )

        avl_exp_dataset = AvalancheDataset(
            exp_dataset,
            transform_groups=base_transforms,
            initial_transform_group='train'
        ).freeze_transforms(
        ).add_transforms_to_group(
            'train', transform=train_transform
        ).add_transforms_to_group(
            'eval', transform=eval_transform
        )

        train_exps.append(avl_exp_dataset)

    # Create the test dataset
    test_exps = []
    exp_dataset = ChallengeDetectionDataset(
        dataset_path,
        train=False,
        ego_api=test_ego_api,
        img_ids=test_img_ids,
    )

    avl_exp_dataset = AvalancheDataset(
        exp_dataset,
        transform_groups=base_transforms,
        initial_transform_group='eval'
    ).freeze_transforms(
    ).add_transforms_to_group(
        'train', transform=train_transform
    ).add_transforms_to_group(
        'eval', transform=eval_transform
    )

    test_exps.append(avl_exp_dataset)

    # Create the validation datasets
    valid_exps = None
    if valid_img_ids is not None:
        valid_exps = []
        for exp_id, exp_img_ids in enumerate(valid_img_ids):
            exp_dataset = ChallengeDetectionDataset(
                dataset_path,
                train=True,
                ego_api=train_ego_api,
                img_ids=exp_img_ids
            )

            avl_exp_dataset = AvalancheDataset(
                exp_dataset,
                transform_groups=base_transforms,
                initial_transform_group='eval'
            ).freeze_transforms(
            ).add_transforms_to_group(
                'train', transform=train_transform
            ).add_transforms_to_group(
                'eval', transform=eval_transform
            )

            valid_exps.append(avl_exp_dataset)

    all_cat_ids = set(train_ego_api.get_cat_ids())
    # all_cat_ids.union(test_ego_api.get_cat_ids())

    stream_definitions = dict()

    train_def = StreamUserDef(
        exps_data=train_exps,
        exps_task_labels=[0 for _ in range(len(train_exps))],
        origin_dataset=None,
        is_lazy=False
    )
    stream_definitions['train'] = train_def

    test_def = StreamUserDef(
        exps_data=test_exps,
        exps_task_labels=[0],
        origin_dataset=None,
        is_lazy=False
    )
    stream_definitions['test'] = test_def

    if valid_exps is not None:
        valid_def = StreamUserDef(
            exps_data=valid_exps,
            exps_task_labels=[0 for _ in range(len(valid_exps))],
            origin_dataset=None,
            is_lazy=False
        )
        stream_definitions['valid'] = valid_def

    return DetectionCLScenario(
        n_classes=len(all_cat_ids),
        stream_definitions=stream_definitions,
        complete_test_set_only=True
    )


def challenge_instance_detection_benchmark(
        dataset_path: Union[str, Path],
        *,
        class_order_seed: int = DEFAULT_CHALLENGE_CLASS_ORDER_SEED,
        train_transform=None,
        eval_transform=None,
        train_json_name=None,
        test_json_name=None,
        n_exps=CHALLENGE_DETECTION_EXPERIENCES,
        n_validation_videos=0,
        validation_video_selection_seed=1337):
    """
    Creates the challenge instance detection benchmark.

    Please don't change this code. You are free to customize the dataset
    path and jsons paths, the train_transform, and eval_transform parameters.
    Don't change other parameters or the code.

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
    :param n_exps: The number of experiences in the training set.
    :param n_validation_videos: How many validation videos per instance.
        Defaults to 0, which means that no validation stream will be created.
    :param validation_video_selection_seed: The seed to use when selecting
        videos to allocate to the validation stream.
    :return: The instance detection benchmark.
    """

    if train_json_name is None:
        train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON

    if test_json_name is None:
        test_json_name = DEFAULT_CHALLENGE_TEST_JSON

    # Use the classification benchmark creator to generate the correct order
    cls_benchmark: NCScenario = challenge_classification_benchmark(
        dataset_path,
        class_order_seed=class_order_seed,
        train_json_name=train_json_name,
        test_json_name=test_json_name,
        instance_level=True,
        n_exps=n_exps,
        unlabeled_test_set=False,
        n_validation_videos=n_validation_videos,
        validation_video_selection_seed=validation_video_selection_seed
    )

    # Create aligned datasets
    train_ego_api = EgoObjects(str(Path(dataset_path) / train_json_name))
    test_ego_api = EgoObjects(str(Path(dataset_path) / test_json_name))

    train_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=train_ego_api,
        train=True,
        bbox_margin=20,
        instance_level=True
    )

    test_dataset = ChallengeClassificationDataset(
        dataset_path,
        ego_api=test_ego_api,
        train=False,
        bbox_margin=20,
        instance_level=True
    )

    # Keep this order
    train_order = cls_benchmark.train_exps_patterns_assignment
    test_order = list(itertools.chain.from_iterable(
        cls_benchmark.test_exps_patterns_assignment))

    train_img_ids = []
    for exp_id in range(len(train_order)):
        img_id_in_exp = []
        for instance_idx in train_order[exp_id]:
            img_id_in_exp.append(train_dataset.img_ids[instance_idx])
        train_img_ids.append(img_id_in_exp)

    test_img_ids = []
    for instance_idx in test_order:
        test_img_ids.append(test_dataset.img_ids[instance_idx])

    valid_img_ids = None
    if 'valid' in cls_benchmark.streams:
        valid_order = cls_benchmark.valid_exps_patterns_assignment
        valid_img_ids = []
        for exp_id in range(len(valid_order)):
            img_id_in_exp = []
            for instance_idx in valid_order[exp_id]:
                img_id_in_exp.append(train_dataset.img_ids[instance_idx])
            valid_img_ids.append(img_id_in_exp)

    base_transforms = dict(
        train=(CHALLENGE_DETECTION_FORCED_TRANSFORMS, None),
        eval=(CHALLENGE_DETECTION_FORCED_TRANSFORMS, None)
    )

    # Remove non-main annotations and replace categories with instances.
    # Considering that the train and test set contain the same set of instances
    # (main objects), the resulting category IDs will be aligned.
    make_instance_based(
        train_ego_api
    )

    # This should not remove annotations (as test annotations only contain the
    # main object bounding box).
    make_instance_based(
        test_ego_api
    )

    # Create the training datasets
    train_exps = []
    for exp_id, exp_img_ids in enumerate(train_img_ids):
        exp_dataset = ChallengeDetectionDataset(
            dataset_path,
            train=True,
            ego_api=train_ego_api,
            img_ids=exp_img_ids,
        )

        avl_exp_dataset = AvalancheDataset(
            exp_dataset,
            transform_groups=base_transforms,
            initial_transform_group='train'
        ).freeze_transforms(
        ).add_transforms_to_group(
            'train', transform=train_transform
        ).add_transforms_to_group(
            'eval', transform=eval_transform
        )

        train_exps.append(avl_exp_dataset)

    # Create the test dataset
    test_exps = []
    exp_dataset = ChallengeDetectionDataset(
        dataset_path,
        train=False,
        ego_api=test_ego_api,
        img_ids=test_img_ids,
    )

    avl_exp_dataset = AvalancheDataset(
        exp_dataset,
        transform_groups=base_transforms,
        initial_transform_group='eval'
    ).freeze_transforms(
    ).add_transforms_to_group(
        'train', transform=train_transform
    ).add_transforms_to_group(
        'eval', transform=eval_transform
    )

    test_exps.append(avl_exp_dataset)

    # Create the validation datasets
    valid_exps = None
    if valid_img_ids is not None:
        valid_exps = []
        for exp_id, exp_img_ids in enumerate(valid_img_ids):
            exp_dataset = ChallengeDetectionDataset(
                dataset_path,
                train=True,
                ego_api=train_ego_api,
                img_ids=exp_img_ids
            )

            avl_exp_dataset = AvalancheDataset(
                exp_dataset,
                transform_groups=base_transforms,
                initial_transform_group='eval'
            ).freeze_transforms(
            ).add_transforms_to_group(
                'train', transform=train_transform
            ).add_transforms_to_group(
                'eval', transform=eval_transform
            )

            valid_exps.append(avl_exp_dataset)

    all_cat_ids = set(train_ego_api.get_cat_ids())
    # all_cat_ids.union(test_ego_api.get_cat_ids())

    stream_definitions = dict()

    train_def = StreamUserDef(
        exps_data=train_exps,
        exps_task_labels=[0 for _ in range(len(train_exps))],
        origin_dataset=None,
        is_lazy=False
    )
    stream_definitions['train'] = train_def

    test_def = StreamUserDef(
        exps_data=test_exps,
        exps_task_labels=[0],
        origin_dataset=None,
        is_lazy=False
    )
    stream_definitions['test'] = test_def

    if valid_exps is not None:
        valid_def = StreamUserDef(
            exps_data=valid_exps,
            exps_task_labels=[0 for _ in range(len(valid_exps))],
            origin_dataset=None,
            is_lazy=False
        )
        stream_definitions['valid'] = valid_def

    return DetectionCLScenario(
        n_classes=len(all_cat_ids),
        stream_definitions=stream_definitions,
        complete_test_set_only=True
    )


def demo_detection_benchmark(
        dataset_path: Union[str, Path],
        class_order_seed: int,
        **kwargs):
    if 'n_exps' not in kwargs:
        kwargs['n_exps'] = DEMO_DETECTION_EXPERIENCES

    if 'train_json_name' not in kwargs:
        kwargs['train_json_name'] = DEFAULT_DEMO_TRAIN_JSON

    if 'test_json_name' not in kwargs:
        kwargs['test_json_name'] = DEFAULT_DEMO_TEST_JSON

    warnings.warn('You are using the demo benchmark. For the competition, '
                  'please use challenge_category_detection_benchmark instead.')

    return challenge_category_detection_benchmark(
        dataset_path=dataset_path,
        class_order_seed=class_order_seed,
        **kwargs
    )


__all__ = [
    'challenge_category_detection_benchmark',
    'challenge_instance_detection_benchmark',
    'demo_detection_benchmark'
]
