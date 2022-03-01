from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.utils import PathsDataset
from ego_objectron import EgoObjectron


class EgoObjectronClassificationDataset(PathsDataset):
    def __init__(
            self,
            root: Union[str, Path],
            ego_api: EgoObjectron,
            img_ids: List[int] = None,
            transform=None,
            target_transform=None,
            loader=default_loader):
        if img_ids is None:
            img_ids = list(sorted(ego_api.get_img_ids()))

        image_triplets, img_ids = self.get_main_instances(
            ego_api, img_ids
        )

        for image_triplet in image_triplets:
            img_dict = image_triplet[0]
            img_url = img_dict['url']
            splitted_url = img_url.split('/')
            img_path = 'cltest/' + splitted_url[-1]
            image_triplet[0] = img_path

        super(EgoObjectronClassificationDataset, self).__init__(
            root=root,
            files=image_triplets,
            transform=transform,
            target_transform=target_transform,
            loader=loader
        )

    @staticmethod
    def get_main_instances(ego_api: EgoObjectron, img_ids: List[int]):
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

            main_bbox = main_annotation['bbox']  # Assume L, T, W, H
            # However, PathDataset requires top, left, height, width
            tmp = main_bbox[0]
            main_bbox[0] = main_bbox[1]
            main_bbox[1] = tmp
            tmp = main_bbox[2]
            main_bbox[2] = main_bbox[3]
            main_bbox[3] = tmp

            image_triplets.append([img_dict, main_annotation['instance_id'], main_bbox])
            all_instance_ids.add(main_annotation['instance_id'])

        class_label_to_instance_id = sorted(list(all_instance_ids))

        # Map from instance_id to class label
        for img_triplet in image_triplets:
            img_triplet[1] = class_label_to_instance_id.index(img_triplet[1])

        return image_triplets, img_ids_with_main_ann


if __name__ == '__main__':
    sample_root: Path = Path.home() / '3rd_clvision_challenge'

    ego_api = EgoObjectron(str(sample_root / "egoobjects_test.json"))
    sample_classification_dataset = EgoObjectronClassificationDataset(
        root=sample_root,
        ego_api=ego_api,
    )

    print(
        'The dataset contains',
        len(set(sample_classification_dataset.targets)),
        'main objects')

    print('Dataset len:', len(sample_classification_dataset))

    n_to_show = 5
    for img_idx in range(n_to_show):
        image, label = sample_classification_dataset[img_idx]
        plt.title(f'Class label: {label}')
        plt.imshow(image)
        plt.show()
        plt.clf()


__all__ = [
    'EgoObjectronClassificationDataset'
]
