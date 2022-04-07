from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict

import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.utils import PathsDataset
from devkit_tools.challenge_constants import DEFAULT_CHALLENGE_TRAIN_JSON, \
    DEFAULT_CHALLENGE_TEST_JSON
from ego_objects import EgoObjects, EgoObjectsImage


class ChallengeClassificationDataset(PathsDataset):
    def __init__(
            self,
            root: Union[str, Path],
            ego_api: EgoObjects = None,
            img_ids: List[int] = None,
            train=True,
            bbox_margin: Union[float, int] = 0,
            transform=None,
            target_transform=None,
            loader=default_loader,
            *,
            instance_level=True):

        self.train = train
        root = Path(root)

        if ego_api is None:
            if self.train:
                ann_json_path = str(root / DEFAULT_CHALLENGE_TRAIN_JSON)
            else:
                ann_json_path = str(root / DEFAULT_CHALLENGE_TEST_JSON)
            ego_api = EgoObjects(ann_json_path)

        if img_ids is None:
            img_ids = list(sorted(ego_api.get_img_ids()))

        image_triplets, self.img_ids = self.get_main_instances(
            ego_api, img_ids, instance_level=instance_level
        )

        # Enlarge bounding box (to include some background in the image)
        if bbox_margin > 0:
            for image_triplet in image_triplets:
                bbox = image_triplet[2]
                img_dict: EgoObjectsImage = image_triplet[0]

                if isinstance(bbox_margin, int):
                    bbox_margin_w = bbox_margin
                    bbox_margin_h = bbox_margin
                else:  # Float number
                    bbox_margin_w = int(img_dict['width'] * bbox_margin)
                    bbox_margin_h = int(img_dict['height'] * bbox_margin)

                max_bbox_margin_h = (img_dict['height'] - (bbox[0] + bbox[2]))
                max_bbox_margin_w = (img_dict['width'] - (bbox[1] + bbox[3]))
                bbox[2] += bbox_margin_h
                bbox[3] += bbox_margin_w
                bbox[2] += min(bbox_margin_h, max_bbox_margin_h)  # Height
                bbox[3] += min(bbox_margin_w, max_bbox_margin_w)  # Width

                bbox[0] = max(bbox[0] - bbox_margin_h, 0)  # Top
                bbox[1] = max(bbox[1] - bbox_margin_w, 0)  # Left

        for image_triplet in image_triplets:
            img_dict = image_triplet[0]
            img_url = img_dict['url']
            splitted_url = img_url.split('/')
            img_path = 'images/' + splitted_url[-1]
            if not (root / img_path).exists():
                img_path = 'cltest/' + splitted_url[-1]
            image_triplet[0] = img_path

        super(ChallengeClassificationDataset, self).__init__(
            root=root,
            files=image_triplets,
            transform=transform,
            target_transform=target_transform,
            loader=loader
        )

    @staticmethod
    def get_main_instances(
            ego_api: EgoObjects,
            img_ids: List[int],
            *,
            instance_level: bool = True):
        image_triplets = []
        all_instance_ids = set()
        img_ids_with_main_ann = []
        for img_id in img_ids:
            img_dict = ego_api.load_imgs(ids=[img_id])[0]

            # img_ids_with_main_ann
            main_annotations = img_dict['main_category_instance_ids']
            if len(main_annotations) != 1:
                continue
            img_ids_with_main_ann.append(img_id)

            main_annotation_id = main_annotations[0]
            main_annotation = ego_api.load_anns(ids=[main_annotation_id])[0]

            main_bbox = list(main_annotation['bbox'])  # Assume L, T, W, H
            # However, PathDataset requires top, left, height, width
            tmp = main_bbox[0]
            main_bbox[0] = main_bbox[1]
            main_bbox[1] = tmp
            tmp = main_bbox[2]
            main_bbox[2] = main_bbox[3]
            main_bbox[3] = tmp

            if instance_level:
                main_annotation_class = main_annotation['instance_id']
            else:
                main_annotation_class = main_annotation['category_id']

            image_triplets.append(
                [img_dict,
                 main_annotation_class,
                 main_bbox])

            all_instance_ids.add(main_annotation_class)

        class_label_to_instance_id = list(sorted(all_instance_ids))
        reversed_mapping = dict()
        for mapped_id, real_id in enumerate(class_label_to_instance_id):
            reversed_mapping[real_id] = mapped_id

        # Map from instance_id to class label
        for img_triplet in image_triplets:
            img_triplet[1] = reversed_mapping[img_triplet[1]]

        return image_triplets, img_ids_with_main_ann

    @staticmethod
    def class_to_videos(
            ego_api: EgoObjects,
            img_ids: List[int],
            *,
            instance_level: bool = True):
        classes_to_videos: \
            Dict[Union[int, str], Dict[int, List[int]]] = \
            defaultdict(lambda: defaultdict(list))

        all_instance_ids = set()

        for img_id in img_ids:
            img_dict = ego_api.load_imgs(ids=[img_id])[0]

            main_annotations = img_dict['main_category_instance_ids']
            if len(main_annotations) != 1:
                continue

            main_annotation_id = main_annotations[0]
            main_annotation = ego_api.load_anns(ids=[main_annotation_id])[0]

            if instance_level:
                main_annotation_class = main_annotation['instance_id']
            else:
                main_annotation_class = main_annotation['category_id']

            classes_to_videos[main_annotation_class][img_dict[
                'gaia_id']].append(img_id)

            all_instance_ids.add(main_annotation_class)

        class_label_to_instance_id = list(sorted(all_instance_ids))
        reversed_mapping = dict()
        for mapped_id, real_id in enumerate(class_label_to_instance_id):
            reversed_mapping[real_id] = mapped_id

        remapped_dict = dict()
        for cls_orig_key, cls_videos in classes_to_videos.items():
            remapped_key = reversed_mapping[cls_orig_key]
            remapped_dict[remapped_key] = cls_videos

        return remapped_dict


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Resize, Compose

    sample_root: Path = Path.home() / '3rd_clvision_challenge' / 'demo_dataset'
    show_images = True
    try_loading = False
    train = True
    instance_level = True

    sample_classification_dataset = ChallengeClassificationDataset(
        root=sample_root,
        train=train,
        bbox_margin=20,
        instance_level=instance_level
    )

    targets = torch.tensor(list(sample_classification_dataset.targets))
    unique_targets, targets_count = torch.unique(targets, return_counts=True)

    print('The dataset contains', len(unique_targets), 'main objects')

    print('Dataset len:', len(sample_classification_dataset))
    for t, t_c in zip(unique_targets, targets_count):
        print('Class', int(t), 'has', int(t_c), 'instances')

    if show_images:
        n_to_show = 5
        for img_idx in range(n_to_show):
            image, label = sample_classification_dataset[img_idx]
            plt.title(f'Class label: {label}')
            plt.imshow(image)
            plt.show()
            plt.clf()

    if try_loading:
        sample_classification_dataset.transform = Compose(
            [Resize((224, 224)), ToTensor()]
        )

        loader = DataLoader(
            sample_classification_dataset,
            batch_size=5,
            num_workers=4)

        for x, y in loader:
            print('.', end='', flush=True)


__all__ = [
    'ChallengeClassificationDataset'
]
