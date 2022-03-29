from typing import List

from ego_objects import EgoObjects, EgoObjectsJson


def make_compact_category_ids_mapping(
        ego_api: EgoObjects, test_ego_api=None) -> List[int]:
    train_category_ids = set(ego_api.get_cat_ids())

    if test_ego_api is not None:
        if train_category_ids != set(test_ego_api.get_cat_ids()):
            raise ValueError(
                'Train and test datasets must contain the same categories!')

    # 0 is added to consider the background ID
    return [0] + list(sorted(train_category_ids))


def remap_category_ids(ego_api: EgoObjects, categories_id_mapping: List[int]):
    """
    Remaps the category IDs by modifying the API object in-place.

    :param ego_api: The API object to adapt.
    :param categories_id_mapping: The category mapping. It must define a
        mapping from the to-be-used-id to the real category id so that:
        `real_cat_id = categories_id_mapping[mapped_id]`.
    """
    reversed_mapping = dict()
    for mapped_id, real_id in enumerate(categories_id_mapping):
        reversed_mapping[real_id] = mapped_id

    dataset_json: EgoObjectsJson = ego_api.dataset

    for cat_dict in dataset_json['categories']:
        cat_dict['id'] = reversed_mapping[cat_dict['id']]

    for ann_dict in dataset_json['annotations']:
        ann_dict['category_id'] = reversed_mapping[ann_dict['category_id']]

    ego_api.recreate_index()


__all__ = [
    'make_compact_category_ids_mapping',
    'remap_category_ids'
]
