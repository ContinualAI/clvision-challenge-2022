import copy
import itertools
import logging
from typing import List

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist

from ego_objects import EgoObjects, EgoObjectsResults, EgoObjectsEval


class EgoEvaluator:
    def __init__(self, ego_gt: EgoObjects, iou_types: List[str]):
        assert isinstance(iou_types, (list, tuple))
        self.ego_gt = ego_gt
        self.iou_types = iou_types
        self.img_ids = []
        self.predictions = []
        self.ego_eval_per_iou = dict()

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare_for_ego_detection(predictions)
        self.predictions.extend(results)

    def synchronize_between_processes(self):
        if dist.is_initialized():
            # Bypass NCCL (which forces CUDA-only sync)
            if dist.get_backend() == "nccl":
                group = dist.new_group(backend="gloo")
            else:
                group = dist.group.WORLD

            my_rank = dist.get_rank()
            output = [None for _ in range(dist.get_world_size())]
            dist.gather_object(
                self.predictions,
                output if my_rank == 0 else None,
                dst=0,
                group=group)

            return list(itertools.chain.from_iterable(output)), my_rank == 0
        else:
            return self.predictions, True

    def evaluate(self, max_dets_per_image=None):
        all_preds, main_process = self.synchronize_between_processes()
        if main_process:
            if max_dets_per_image is None:
                max_dets_per_image = 300

            eval_imgs = [ego_res['image_id'] for ego_res in all_preds]

            gt_subset = EgoEvaluator._make_ego_subset(self.ego_gt, eval_imgs)

            for iou_type in self.iou_types:
                print('Evaluating for iou', iou_type)
                if iou_type == "segm":
                    # See:
                    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/lvis_evaluation.html
                    ego_results = copy.deepcopy(all_preds)
                    for c in ego_results:
                        c.pop("bbox", None)
                else:
                    ego_results = all_preds

                ego_results = EgoObjectsResults(
                    gt_subset, ego_results, max_dets=max_dets_per_image)
                ego_eval = EgoObjectsEval(gt_subset, ego_results, iou_type)
                ego_eval.params.img_ids = list(set(eval_imgs))
                ego_eval.run()
                self.ego_eval_per_iou[iou_type] = ego_eval
        else:
            self.ego_eval_per_iou = None

        if dist.is_initialized():
            dist.barrier()

        return self.ego_eval_per_iou

    def summarize(self):
        if self.ego_eval_per_iou is not None:
            for iou_type, ego_eval in self.ego_eval_per_iou.items():
                print(f"IoU metric: {iou_type}")
                ego_eval.print_results()

    def prepare_for_ego_detection(self, predictions):
        ego_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            has_mask = 'mask' in prediction
            has_bbox = 'boxes' in prediction
            has_keypoint = 'keypoints' in prediction

            if has_bbox:
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()

            if has_mask:
                masks = prediction["masks"]
                masks = masks > 0.5
                rles = [
                    mask_util.encode(
                        np.array(mask[0, :, :, np.newaxis], dtype=np.uint8,
                                 order="F"))[0] for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

            if has_keypoint:
                keypoints = prediction["keypoints"]
                keypoints = keypoints.flatten(start_dim=1).tolist()

            for pred_idx in range(len(labels)):
                ego_pred = {
                    "image_id": original_id,
                    "category_id": labels[pred_idx],
                    "score": scores[pred_idx],
                }

                if has_bbox:
                    ego_pred["bbox"] = boxes[pred_idx]

                if has_mask:
                    ego_pred['segmentation'] = rles[pred_idx]

                if has_keypoint:
                    ego_pred['keypoints'] = keypoints[pred_idx]

                ego_results.append(ego_pred)

        return ego_results

    @staticmethod
    def _make_ego_subset(ego_gt, img_ids):
        img_ids = set(img_ids)

        subset = dict()
        subset['categories'] = list(ego_gt.dataset["categories"])

        subset_imgs = []
        for img in ego_gt.dataset["images"]:
            if img["id"] in img_ids:
                subset_imgs.append(img)
        subset['images'] = subset_imgs

        subset_anns = []
        for ann in ego_gt.dataset["annotations"]:
            if ann["image_id"] in img_ids:
                subset_anns.append(ann)
        subset['annotations'] = subset_anns

        return DictEgo(subset)


class DictEgo(EgoObjects):
    """
    Child class of EgoObjects that allows for the creation of EgoObjects
    objects from a dictionary.
    """

    def __init__(self, annotation_dict):
        """Class for reading and visualizing annotations.
        Args:
            annotation_dict (dict): annotations
        """
        self.logger = logging.getLogger(__name__)
        self.dataset = annotation_dict

        assert (
                type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(
            type(self.dataset))
        self._create_index()


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
