"""
API for accessing EgoObjects Dataset.

EgoObjects API is a Python API that assists in loading, parsing and visualizing
the annotations in EgoObjects. In addition to this API, please download
images and annotations from the EgoObjects website.
"""

import json
import os
import logging
from collections import defaultdict
from typing import Dict, TypeVar, List, Iterable
from urllib.request import urlretrieve

import pycocotools.mask as mask_utils

from ego_objects import EgoObjectsAnnotation, EgoObjectsCategory, \
    EgoObjectsImage, EgoObjectsJson

T = TypeVar('T')

BAD_ANNS = [127172]


class EgoObjects:
    def __init__(self, annotation_path, annotation_dict=None):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): Location of annotation file
            annotation_dict (dict): Already loaded annotations.
                If set, overrides annotation_path.
        """
        self.logger = logging.getLogger(__name__)

        self.dataset: EgoObjectsJson

        if annotation_dict is None:
            self.logger.info("Loading annotations.")
            self.dataset = self._load_json(annotation_path)
        else:
            self.logger.info("Using pre-loaded annotations.")
            self.dataset = annotation_dict

        assert (
                type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))

        self._fix_bboxes()
        self._fix_areas(force_recompute=False)
        self._fix_frequencies()
        self._create_index()

    def _fix_bboxes(self):
        to_remove_ann_indices = list()
        for ann_idx, ann in enumerate(self.dataset["annotations"]):
            ann_id = ann['id']
            if ann_id in BAD_ANNS:
                to_remove_ann_indices.append(ann_idx)

        for to_remove_idx in sorted(to_remove_ann_indices, reverse=True):
            print('Removed bad annotation',
                  self.dataset["annotations"][to_remove_idx]['id'])
            del self.dataset["annotations"][to_remove_idx]

    def _fix_areas(self, force_recompute=False):
        for ann in self.dataset["annotations"]:
            if force_recompute:
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]
            else:
                ann['area'] = ann.get('area', ann['bbox'][2] * ann['bbox'][3])

    def _fix_frequencies(self):
        for cat_data in self.dataset["categories"]:
            if 'frequency' not in cat_data:
                # r: Rare    :  < 10
                # c: Common  : >= 10 and < 100
                # f: Frequent: >= 100
                if cat_data['image_count'] < 10:
                    frequency = 'r'
                elif cat_data['image_count'] < 100:
                    frequency = 'c'
                else:
                    frequency = 'f'
                cat_data['frequency'] = frequency

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        self.logger.info("Creating index.")

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns: Dict[int, EgoObjectsAnnotation] = {}
        self.cats: Dict[int, EgoObjectsCategory] = {}
        self.imgs: Dict[int, EgoObjectsImage] = {}

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        self.logger.info("Index created.")

    def recreate_index(self):
        self._create_index()

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None) -> List[int]:
        """Get ann ids that satisfy given filter conditions.

        Args:
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range. e.g [0, inf]

        Returns:
            ids (int array): integer array of ann ids
        """
        anns: List[EgoObjectsAnnotation] = []
        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset["annotations"]

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann["id"] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float("inf")]

        ann_ids = []
        for _ann in anns:
            ann_area = _ann.get(
                'area', _ann['bbox'][2] * _ann['bbox'][3])
            if _ann["category_id"] in cat_ids \
                    and area_rng[0] < ann_area < area_rng[1]:
                ann_ids.append(_ann["id"])

        # ann_ids = [
        #     _ann["id"]
        #     for _ann in anns
        #     if _ann["category_id"] in cat_ids
        #     and _ann["area"] > area_rng[0]
        #     and _ann["area"] < area_rng[1]
        # ]
        return ann_ids

    def get_cat_ids(self) -> List[int]:
        """Get all category ids.

        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())

    def get_img_ids(self) -> List[int]:
        """Get all img ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict: Dict[int, T], ids: Iterable[int]) -> List[T]:
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, ids=None) -> List[EgoObjectsAnnotation]:
        """Load anns with the specified ids. If ids=None load all anns.

        Args:
            ids (int array): integer array of annotation ids

        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_cats(self, ids) -> List[EgoObjectsCategory]:
        """Load categories with the specified ids. If ids=None load all
        categories.

        Args:
            ids (int array): integer array of category ids

        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids) -> List[EgoObjectsImage]:
        """Load categories with the specified ids. If ids=None load all images.

        Args:
            ids (int array): integer array of image ids

        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    def download(self, save_dir, img_ids=None):
        """Download images from mscoco.org server.
        Args:
            save_dir (str): dir to save downloaded images
            img_ids (int array): img ids of images to download
        """
        raise RuntimeError('On-the-fly download is not available yet.')

        # imgs = self.load_imgs(img_ids)
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # for img in imgs:
        #     file_name = os.path.join(save_dir, img["coco_url"].split("/")[-1])
        #     if not os.path.exists(file_name):
        #         urlretrieve(img["coco_url"], file_name)

    def ann_to_rle(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.
        Args:
            ann (dict) : annotation object

        Returns:
            ann (rle)
        """
        img_data = self.imgs[ann["image_id"]]
        h, w = img_data["height"], img_data["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def ann_to_mask(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object

        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann)
        return mask_utils.decode(rle)
