import itertools
from pathlib import Path
from typing import Union

from avalanche.benchmarks import NCScenario, StreamUserDef
from avalanche.benchmarks.utils import AvalancheDataset
from devkit_tools import ChallengeClassificationDataset, \
    ChallengeDetectionDataset
from devkit_tools.benchmarks.classification_benchmark import \
    demo_classification_benchmark
from devkit_tools.benchmarks.detection_scenario import DetectionCLScenario, \
    det_exp_factory
from devkit_tools.challenge_constants import \
    DEMO_DETECTION_FORCED_TRANSFORMS, DEFAULT_DEMO_TEST_JSON, \
    DEFAULT_DEMO_TRAIN_JSON, DEMO_DETECTION_EXPERIENCES
from ego_objectron import EgoObjectron


def demo_detection_benchmark(
        dataset_path: Union[str, Path],
        class_order_seed: int,
        *,
        train_transform=None,
        eval_transform=None,
        train_json_name=None,
        test_json_name=None):

    # Use the classification benchmark creator to generate the correct order
    cls_benchmark: NCScenario = demo_classification_benchmark(
        dataset_path,
        class_order_seed,
        train_json_name=train_json_name,
        test_json_name=test_json_name,
        instance_level=False,
        n_exps=DEMO_DETECTION_EXPERIENCES
    )

    # Create aligned datasets
    if train_json_name is None:
        train_json_name = DEFAULT_DEMO_TRAIN_JSON
    train_ego_api = EgoObjectron(str(Path(dataset_path) / train_json_name))

    if test_json_name is None:
        test_json_name = DEFAULT_DEMO_TEST_JSON
    test_ego_api = EgoObjectron(str(Path(dataset_path) / test_json_name))

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

    base_transforms = dict(
        train=(DEMO_DETECTION_FORCED_TRANSFORMS, None),
        eval=(DEMO_DETECTION_FORCED_TRANSFORMS, None)
    )

    # Align categories IDs
    # The JSON may contain categories with sparse IDs
    # In this way max(categories_ids) >= len(categories), which is not ok!
    # For instance, if category IDs are [0, 1, 2, 3, 5], then initializing
    # the ROI head with n_categories=5 is wrong and it will trigger errors
    # when computing losses (as logits must have 6 elements, not 5)
    # To prevent issues, we just remap categories IDs to range [0, n_categories)
    train_category_ids = set(train_ego_api.get_cat_ids())
    assert train_category_ids == set(test_ego_api.get_cat_ids())

    categories_id_mapping = list(sorted(train_category_ids))

    train_exps = []
    for exp_id, exp_img_ids in enumerate(train_img_ids):
        exp_dataset = ChallengeDetectionDataset(
            dataset_path,
            train=True,
            ego_api=train_ego_api,
            img_ids=exp_img_ids,
            categories_id_mapping=categories_id_mapping
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

    test_exps = []
    exp_dataset = ChallengeDetectionDataset(
        dataset_path,
        train=False,
        ego_api=test_ego_api,
        img_ids=test_img_ids,
        categories_id_mapping=categories_id_mapping
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

    all_cat_ids = set(train_ego_api.get_cat_ids())
    all_cat_ids.union(test_ego_api.get_cat_ids())

    train_def = StreamUserDef(
        exps_data=train_exps,
        exps_task_labels=[0 for _ in range(len(train_exps))],
        origin_dataset=None,
        is_lazy=False
    )

    test_def = StreamUserDef(
        exps_data=test_exps,
        exps_task_labels=[0],
        origin_dataset=None,
        is_lazy=False
    )

    return DetectionCLScenario(
        n_classes=len(all_cat_ids),
        stream_definitions={
            'train': train_def,
            'test': test_def
        },
        complete_test_set_only=True,
        experience_factory=det_exp_factory
    )


__all__ = [
    'demo_detection_benchmark'
]
