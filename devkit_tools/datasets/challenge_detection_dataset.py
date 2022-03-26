###
# Adapted from Avalanche LvisDataset
# https://github.com/ContinualAI/avalanche/tree/detection/avalanche/benchmarks/datasets/lvis
#
# Released under the MIT license, see:
# https://github.com/ContinualAI/avalanche/blob/master/LICENSE
###

from pathlib import Path
from typing import List, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from devkit_tools.challenge_constants import DEFAULT_CHALLENGE_TRAIN_JSON, \
    DEFAULT_CHALLENGE_TEST_JSON
from ego_objects import EgoObjects, EgoObjectsAnnotation, \
    EgoObjectsImage
import torch


class ChallengeDetectionDataset(Dataset):
    """
    The sample dataset. For internal use by challenge organizers only.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        train=True,
        transform=None,
        loader=default_loader,
        ego_api=None,
        img_ids: List[int] = None,
        bbox_format: str = 'ltwh',
        categories_id_mapping: List[int] = None
    ):
        """
        Instantiates the sample dataset.

        :param root: The path to the images and annotation file.
        :param transform: The transformation to apply.
        :param loader: The image loader. Defaults to PIL Image open.
        :param ego_api: An EgoObjects object. If not provided, annotations
            will be loaded from the json file found in the root. Defaults to
            None.
        :param img_ids: A list of image ids to use. If not None, only those
            images (a subset of the original dataset) will be used. Defaults
            to None.
        :param bbox_format: The bounding box format. Defaults to "ltwh"
            (Left, Top, Width, Height).
        :param categories_id_mapping: If set, it must define a mapping from
            the to-be-used-id to the real category id so that:
            real_cat_id = categories_id_mapping[mapped_id].
        """
        self.root: Path = Path(root)
        self.train = train
        self.transform = transform
        self.loader = loader
        self.bbox_crop = True
        self.img_ids = img_ids
        self.bbox_format = bbox_format
        self.categories_id_mapping = categories_id_mapping

        self.ego_api = ego_api

        must_load_api = self.ego_api is None
        must_load_img_ids = self.img_ids is None

        # Load metadata
        if must_load_api:
            if self.train:
                ann_json_path = str(self.root / DEFAULT_CHALLENGE_TRAIN_JSON)
            else:
                ann_json_path = str(self.root / DEFAULT_CHALLENGE_TEST_JSON)
            self.ego_api = EgoObjects(ann_json_path)

        if must_load_img_ids:
            self.img_ids = list(sorted(self.ego_api.get_img_ids()))

        self.targets = EgoObjectsDetectionTargets(
            self.ego_api, self.img_ids,
            categories_id_mapping=categories_id_mapping)

        # Try loading an image
        if len(self.img_ids) > 0:
            img_id = self.img_ids[0]
            img_dict = self.ego_api.load_imgs(ids=[img_id])[0]
            assert self._load_img(img_dict) is not None

    def __getitem__(self, index):
        """
        Loads an instance given its index.

        :param index: The index of the instance to retrieve.

        :return: a (sample, target) tuple where the target is a
            torchvision-style annotation for object detection
            https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """
        img_id = self.img_ids[index]
        img_dict: EgoObjectsImage = self.ego_api.load_imgs(ids=[img_id])[0]
        annotation_dicts = self.targets[index]

        # Transform from EgoObjects dictionary to torchvision-style target
        num_objs = len(annotation_dicts)

        boxes = []
        labels = []
        areas = []
        for i in range(num_objs):
            xmin = annotation_dicts[i]['bbox'][0]
            ymin = annotation_dicts[i]['bbox'][1]
            if self.bbox_format == 'ltrb':
                # Left, Top, Right, Bottom
                xmax = annotation_dicts[i]['bbox'][2]
                ymax = annotation_dicts[i]['bbox'][3]

                boxw = xmax - xmin
                boxh = ymax - ymin
            else:
                # Left, Top, Width, Height
                boxw = annotation_dicts[i]['bbox'][2]
                boxh = annotation_dicts[i]['bbox'][3]

                xmax = boxw + xmin
                ymax = boxh + ymin

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation_dicts[i]['category_id'])
            areas.append(boxw * boxh)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        img = self._load_img(img_dict)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def _load_img(self, img_dict):
        img_url = img_dict['url']
        splitted_url = img_url.split('/')
        img_path = 'images/' + splitted_url[-1]
        img_path_alt = 'cltest/' + splitted_url[-1]

        final_path = self.root / img_path  # <root>/images/<img_id>.jpg
        if not final_path.exists():
            final_path = self.root / img_path_alt
        return Image.open(str(final_path)).convert("RGB")


class EgoObjectsDetectionTargets(Sequence[List[EgoObjectsAnnotation]]):
    def __init__(
            self,
            ego_api: EgoObjects,
            img_ids: List[int] = None,
            categories_id_mapping: List[int] = None):
        super(EgoObjectsDetectionTargets, self).__init__()
        self.ego_api = ego_api

        if categories_id_mapping is not None:
            self.reversed_mapping = dict()
            for mapped_id, real_id in enumerate(categories_id_mapping):
                self.reversed_mapping[real_id] = mapped_id
        else:
            self.reversed_mapping = None

        if img_ids is None:
            img_ids = list(sorted(ego_api.get_img_ids()))

        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        annotation_ids = self.ego_api.get_ann_ids(img_ids=[img_id])
        annotation_dicts: List[EgoObjectsAnnotation] = \
            self.ego_api.load_anns(annotation_ids)

        if self.reversed_mapping is None:
            return annotation_dicts

        mapped_anns: List[EgoObjectsAnnotation] = []
        for ann_dict in annotation_dicts:
            ann_dict: EgoObjectsAnnotation = dict(ann_dict)
            ann_dict['category_id'] = \
                self.reversed_mapping[ann_dict['category_id']]
            mapped_anns.append(ann_dict)

        return mapped_anns


__all__ = [
    'ChallengeDetectionDataset'
]
