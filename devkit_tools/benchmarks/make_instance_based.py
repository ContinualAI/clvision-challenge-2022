from collections import defaultdict

from ego_objects import EgoObjects, EgoObjectsJson


def make_instance_based(ego_api: EgoObjects):
    main_annotations_ids = set()
    main_annotations_dicts = []
    unique_object_ids = set()

    ego_dataset: EgoObjectsJson = ego_api.dataset
    for img_dict in ego_dataset['images']:
        main_category_instance_ids = img_dict['main_category_instance_ids']
        assert len(main_category_instance_ids) == 1

        main_annotations_ids.add(main_category_instance_ids[0])

    for ann_dict in ego_dataset['annotations']:
        if ann_dict['id'] in main_annotations_ids:
            main_annotations_dicts.append(ann_dict)
            unique_object_ids.add(ann_dict['instance_id'])
    ego_dataset['annotations'] = main_annotations_dicts

    unique_object_ids_sorted = ['background'] + list(sorted(unique_object_ids))
    reversed_mapping = dict()
    for mapped_id, real_id in enumerate(unique_object_ids_sorted):
        reversed_mapping[real_id] = mapped_id

    img_count = defaultdict(int)
    for ann_dict in ego_dataset['annotations']:
        inst_id = ann_dict['instance_id']
        new_id = reversed_mapping[inst_id]
        ann_dict['category_id'] = new_id
        img_count[new_id] += 1

    new_categories = []

    for cat_id in unique_object_ids_sorted[1:]:  # Exclude the background
        new_cat_dict = dict(
            id=reversed_mapping[cat_id],
            name=f'Object{cat_id}',
            image_count=img_count[cat_id],
            instance_count=img_count[cat_id]
        )
        new_categories.append(new_cat_dict)
    ego_dataset['categories'] = new_categories
    ego_api._fix_frequencies()
    ego_api._create_index()


__all__ = [
    'make_instance_based'
]
